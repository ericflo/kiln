//! Parity test for the production `sdpa_prefill_f32.comp` kernel
//! (`dispatch_sdpa_prefill_f32_bytes`) vs a CPU online-softmax reference.
//!
//! This is the kernel the Vulkan backend's `flash_attn_prefill_vulkan`
//! dispatches for full-attention prefill. It is distinct from the
//! `vk_sdpa_prefill` op exercised by `vk_attention_parity.rs` (that path
//! uses permute + batched-matmul, not this fused online-softmax kernel).
//!
//! Regression guard for the #1082 head_dim=256 fix: the kernel previously
//! hard-assumed head_dim <= 128 (one thread per element) and dropped the
//! upper half of every Q/K/V vector for Qwen3.5-4B (head_dim=256). The
//! grid-stride rewrite (128 threads × 2 elems/thread) must match the CPU
//! reference at head_dim 64 / 128 / 256.

use kiln_vulkan_kernel::VulkanDevice;
use kiln_vulkan_kernel::kernels::dispatch_sdpa_prefill_f32_bytes;
use std::sync::Arc;

fn vk_dev() -> Option<Arc<VulkanDevice>> {
    if !VulkanDevice::probe() {
        return None;
    }
    VulkanDevice::new().ok().map(Arc::new)
}

/// Deterministic pseudo-random-ish fill in a small, well-conditioned range
/// so the F32 softmax stays numerically stable and the parity tolerance is
/// tight.
fn fill(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) % 1000;
            (x as f32 / 1000.0 - 0.5) * 2.0 // in [-1, 1)
        })
        .collect()
}

/// CPU online-softmax reference matching the kernel's contract:
/// token-major `[B, T, H, dh]`, causal, all heads present (no GQA here).
fn cpu_sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let stride_token = num_heads * head_dim;
    let stride_batch = seq_len * stride_token;
    let mut out = vec![0.0_f32; batch * seq_len * num_heads * head_dim];
    for b in 0..batch {
        for h in 0..num_heads {
            for qi in 0..seq_len {
                let q_base = b * stride_batch + qi * stride_token + h * head_dim;
                let k_max = if causal { qi + 1 } else { seq_len };
                // scores
                let mut scores = vec![0.0_f32; k_max];
                let mut max_logit = f32::NEG_INFINITY;
                for ki in 0..k_max {
                    let k_base = b * stride_batch + ki * stride_token + h * head_dim;
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * k[k_base + d];
                    }
                    let logit = dot * scale;
                    scores[ki] = logit;
                    max_logit = max_logit.max(logit);
                }
                let mut sum = 0.0_f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_logit).exp();
                    sum += *s;
                }
                for d in 0..head_dim {
                    let mut acc = 0.0_f32;
                    for ki in 0..k_max {
                        let k_base = b * stride_batch + ki * stride_token + h * head_dim;
                        acc += scores[ki] * v[k_base + d];
                    }
                    out[q_base + d] = acc / sum;
                }
            }
        }
    }
    out
}

fn run_case(
    dev: &VulkanDevice,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    causal: bool,
) {
    let n = batch * seq_len * num_heads * head_dim;
    let q = fill(n, 1);
    let k = fill(n, 7);
    let v = fill(n, 13);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_bytes = bytemuck::cast_slice(&q).to_vec();
    let k_bytes = bytemuck::cast_slice(&k).to_vec();
    let v_bytes = bytemuck::cast_slice(&v).to_vec();

    let out_bytes = dispatch_sdpa_prefill_f32_bytes(
        dev, &q_bytes, &k_bytes, &v_bytes, batch, seq_len, num_heads, head_dim, scale, causal,
    )
    .expect("sdpa_prefill_f32 dispatch");
    let out_vk: &[f32] = bytemuck::cast_slice(&out_bytes);

    let out_cpu = cpu_sdpa(
        &q, &k, &v, batch, seq_len, num_heads, head_dim, scale, causal,
    );

    assert_eq!(out_vk.len(), out_cpu.len());
    let mut max_abs = 0.0_f32;
    for (a, b) in out_vk.iter().zip(out_cpu.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs < 1e-4,
        "sdpa_prefill_f32 parity (B={batch}, T={seq_len}, H={num_heads}, dh={head_dim}, \
         causal={causal}): max_abs_diff {max_abs} exceeds 1e-4"
    );
}

#[test]
fn sdpa_prefill_f32_head_dim_64_matches_cpu() {
    let Some(dev) = vk_dev() else {
        eprintln!("no Vulkan device; skipping");
        return;
    };
    run_case(&dev, 1, 5, 2, 64, true);
    run_case(&dev, 1, 5, 2, 64, false);
}

#[test]
fn sdpa_prefill_f32_head_dim_128_matches_cpu() {
    let Some(dev) = vk_dev() else {
        eprintln!("no Vulkan device; skipping");
        return;
    };
    run_case(&dev, 1, 7, 2, 128, true);
    run_case(&dev, 2, 4, 3, 128, false);
}

/// The Qwen3.5-4B head_dim. Regression guard for the #1082 grid-stride fix.
#[test]
fn sdpa_prefill_f32_head_dim_256_matches_cpu() {
    let Some(dev) = vk_dev() else {
        eprintln!("no Vulkan device; skipping");
        return;
    };
    run_case(&dev, 1, 6, 2, 256, true);
    run_case(&dev, 1, 11, 16, 256, true); // shape from the failing prefill repro
    run_case(&dev, 2, 3, 4, 256, false);
}
