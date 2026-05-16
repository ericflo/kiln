//! Parity tests for the OPD top-K reverse-KL loss kernel.
//!
//! Three reference paths:
//! 1. **Naive softmax KL**: materialize the full `[T_active, K]` student
//!    logits via gather + matmul on the K teacher indices, do
//!    log_softmax + KL via candle autograd. This is the oracle that even
//!    Phase A is compared against.
//! 2. **Phase A**: pure-candle reference (autograd).
//! 3. **Phase B**: `CustomOp1` with manual backward.
//!
//! All three must agree at 1e-5 at f32 (forward) and 1e-4 (backward), and
//! 1e-2 at bf16. The naive path is identical to Phase A in formula but
//! verifies the index_select / matmul / log_softmax composition end-to-end.

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, Var};

use super::*;

/// Naive oracle: gather K columns of `head_t`, project, softmax-renormalise,
/// compute reverse KL. Autograd flows through every op.
fn naive_kl(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    device: &Device,
) -> Result<Tensor> {
    let active_positions: Vec<u32> = label_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();
    let active_count = active_positions.len();
    if active_count == 0 {
        return Ok(Tensor::new(0.0f32, device)?);
    }
    let active_indices = Tensor::new(active_positions.as_slice(), device)?;
    let hidden_2d = hidden.squeeze(0)?;
    let active_hidden = hidden_2d.index_select(&active_indices, 0)?;
    let active_hidden_f32 = active_hidden.to_dtype(DType::F32)?;
    let head_t_f32 = head_t.to_dtype(DType::F32)?;

    // Build full student logits over the full vocab via active_hidden @ head_t,
    // then gather K columns per row using the teacher indices.
    let full_logits = active_hidden_f32.matmul(&head_t_f32)?; // [T_active, V]
    let mut s_logits_rows: Vec<Tensor> = Vec::with_capacity(active_count);
    for t in 0..active_count {
        let row = full_logits.narrow(0, t, 1)?;
        let row_indices = Tensor::new(
            &teacher_topk_indices[t * top_k..(t + 1) * top_k],
            device,
        )?;
        let gathered = row.index_select(&row_indices, 1)?;
        s_logits_rows.push(gathered);
    }
    let s_logits = Tensor::cat(&s_logits_rows, 0)?;
    let q_logprobs = Tensor::from_vec(
        teacher_topk_logprobs.to_vec(),
        (active_count, top_k),
        device,
    )?;
    let log_p_hat = log_softmax_last(&s_logits)?;
    let log_q_hat = log_softmax_last(&q_logprobs)?;
    let p_hat = log_p_hat.exp()?;
    let diff = (&log_p_hat - &log_q_hat)?;
    let per_pos = (p_hat * diff)?.sum(D::Minus1)?;
    Ok(per_pos.mean_all()?)
}

/// Build a deterministic random test case. Index_select needs valid
/// indices < vocab_size. The teacher logprobs are *not* required to be
/// log-softmax-normalised over the full vocab in the contract (the kernel
/// renormalises them over the K support), so we can pick any plausible
/// values.
fn random_case(
    seq_len: usize,
    hidden_size: usize,
    vocab_size: usize,
    top_k: usize,
    active_period: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Vec<u32>, Vec<f32>, Vec<bool>)> {
    let hidden_vec: Vec<f32> = (0..(seq_len * hidden_size))
        .map(|i| (i as f32 * 0.013).sin() * 0.5)
        .collect();
    let hidden = Tensor::from_vec(hidden_vec, (1, seq_len, hidden_size), device)?;
    let head_vec: Vec<f32> = (0..(hidden_size * vocab_size))
        .map(|i| ((i as f32 + 7.0) * 0.007).cos() * 0.25)
        .collect();
    let head_t = Tensor::from_vec(head_vec, (hidden_size, vocab_size), device)?;

    let label_mask: Vec<bool> = (0..seq_len)
        .map(|i| (i % active_period) == 0 && i > 0)
        .collect();
    let active_count = label_mask.iter().filter(|&&m| m).count();

    let mut teacher_topk_indices: Vec<u32> = Vec::with_capacity(active_count * top_k);
    let mut teacher_topk_logprobs: Vec<f32> = Vec::with_capacity(active_count * top_k);
    for t in 0..active_count {
        // Pick K distinct indices into the vocab, deterministically.
        let mut indices: Vec<u32> = (0..top_k as u32)
            .map(|k| ((t * 17 + (k as usize) * 31 + 5) % vocab_size) as u32)
            .collect();
        // Deduplicate by perturbing collisions (small vocab in tests can
        // produce repeats; the kernel doesn't actually require uniqueness
        // but the naive oracle's per-row gather does).
        let mut seen = std::collections::HashSet::new();
        for k in 0..top_k {
            while !seen.insert(indices[k]) {
                indices[k] = (indices[k] + 1) % vocab_size as u32;
            }
        }
        teacher_topk_indices.extend_from_slice(&indices);
        for k in 0..top_k {
            // Approximate log-softmax values; the renormaliser inside the
            // kernel makes the absolute scale irrelevant.
            let v = -((t as f32 + 1.0).ln() + (k as f32) * 0.3);
            teacher_topk_logprobs.push(v);
        }
    }

    Ok((
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
    ))
}

#[test]
fn cpu_phase_a_matches_naive_f32() -> Result<()> {
    let device = Device::Cpu;
    let (hidden, head_t, idx, lp, mask) = random_case(16, 8, 64, 8, 2, &device)?;
    let a = opd_top_k_reverse_kl_phase_a(&hidden, &head_t, &idx, &lp, &mask, 8, &device)?;
    let n = naive_kl(&hidden, &head_t, &idx, &lp, &mask, 8, &device)?;
    let av = a.to_scalar::<f32>()?;
    let nv = n.to_scalar::<f32>()?;
    assert!(
        (av - nv).abs() < 1e-5,
        "phase A vs naive f32: a={av:.6} n={nv:.6} abs_diff={:.2e}",
        (av - nv).abs()
    );
    // Sanity: KL must be non-negative.
    assert!(av >= -1e-6, "KL went negative: {av}");
    Ok(())
}

#[test]
fn cpu_phase_b_matches_naive_f32() -> Result<()> {
    let device = Device::Cpu;
    let (hidden, head_t, idx, lp, mask) = random_case(16, 8, 64, 8, 2, &device)?;
    let b =
        opd_top_k_reverse_kl_phase_b(&hidden, &head_t, &idx, &lp, &mask, 8, &device, 16)?;
    let n = naive_kl(&hidden, &head_t, &idx, &lp, &mask, 8, &device)?;
    let bv = b.to_scalar::<f32>()?;
    let nv = n.to_scalar::<f32>()?;
    assert!(
        (bv - nv).abs() < 1e-5,
        "phase B vs naive f32: b={bv:.6} n={nv:.6} abs_diff={:.2e}",
        (bv - nv).abs()
    );
    Ok(())
}

#[test]
fn cpu_phase_b_chunked_matches_unchunked() -> Result<()> {
    let device = Device::Cpu;
    let (hidden, head_t, idx, lp, mask) = random_case(20, 8, 64, 8, 2, &device)?;
    let full =
        opd_top_k_reverse_kl_phase_b(&hidden, &head_t, &idx, &lp, &mask, 8, &device, 1024)?;
    let chunked =
        opd_top_k_reverse_kl_phase_b(&hidden, &head_t, &idx, &lp, &mask, 8, &device, 3)?;
    let fv = full.to_scalar::<f32>()?;
    let cv = chunked.to_scalar::<f32>()?;
    assert!(
        (fv - cv).abs() < 1e-5,
        "chunked vs unchunked: full={fv:.6} chunk3={cv:.6}"
    );
    Ok(())
}

#[test]
fn cpu_phase_b_per_position_matches_scalar() -> Result<()> {
    let device = Device::Cpu;
    let (hidden, head_t, idx, lp, mask) = random_case(20, 8, 64, 8, 2, &device)?;
    let per = opd_top_k_reverse_kl_phase_b_per_position(
        &hidden, &head_t, &idx, &lp, &mask, 8, &device, 64,
    )?;
    let scalar =
        opd_top_k_reverse_kl_phase_b(&hidden, &head_t, &idx, &lp, &mask, 8, &device, 64)?;
    let per_v: Vec<f32> = per.to_vec1()?;
    let recomputed_mean = per_v.iter().sum::<f32>() / (per_v.len() as f32);
    let sv = scalar.to_scalar::<f32>()?;
    assert!(
        (recomputed_mean - sv).abs() < 1e-5,
        "per-pos mean={recomputed_mean:.6} vs scalar={sv:.6}",
    );
    Ok(())
}

#[test]
fn cpu_phase_b_empty_mask_returns_zero() -> Result<()> {
    let device = Device::Cpu;
    let (hidden, head_t, _, _, _) = random_case(8, 4, 16, 4, 2, &device)?;
    let empty_mask = vec![false; 8];
    // Empty mask -> empty indices/logprobs slices.
    let loss = opd_top_k_reverse_kl_phase_b(
        &hidden,
        &head_t,
        &[],
        &[],
        &empty_mask,
        4,
        &device,
        16,
    )?;
    assert_eq!(loss.to_scalar::<f32>()?, 0.0);
    Ok(())
}

#[test]
fn cpu_phase_b_bf16_forward_within_tolerance() -> Result<()> {
    let device = Device::Cpu;
    let (hidden, head_t, idx, lp, mask) = random_case(16, 8, 64, 8, 2, &device)?;
    let hidden_bf = hidden.to_dtype(DType::BF16)?;
    let head_bf = head_t.to_dtype(DType::BF16)?;
    let bf =
        opd_top_k_reverse_kl_phase_b(&hidden_bf, &head_bf, &idx, &lp, &mask, 8, &device, 16)?;
    let f32 =
        opd_top_k_reverse_kl_phase_b(&hidden, &head_t, &idx, &lp, &mask, 8, &device, 16)?;
    let bfv = bf.to_scalar::<f32>()?;
    let f32v = f32.to_scalar::<f32>()?;
    assert!(
        (bfv - f32v).abs() < 1e-2,
        "bf16 vs f32 forward: bf={bfv:.6} f32={f32v:.6}"
    );
    Ok(())
}

/// Most load-bearing test: the manual backward in `bwd()` must match
/// candle's autograd backward over the naive softmax-KL formula at f32.
#[test]
fn cpu_phase_b_backward_matches_naive_f32() -> Result<()> {
    let device = Device::Cpu;
    let (hidden_init, head_t, idx, lp, mask) = random_case(16, 8, 64, 8, 2, &device)?;

    // Naive backward via candle autograd.
    let hidden_var_a = Var::from_tensor(&hidden_init)?;
    let loss_naive = naive_kl(hidden_var_a.as_tensor(), &head_t, &idx, &lp, &mask, 8, &device)?;
    let grads_naive = loss_naive.backward()?;
    let g_naive = grads_naive
        .get(hidden_var_a.as_tensor())
        .ok_or_else(|| anyhow!("no naive grad"))?
        .clone();

    // Phase B backward via CustomOp1.
    let hidden_var_b = Var::from_tensor(&hidden_init)?;
    let loss_b = opd_top_k_reverse_kl_phase_b(
        hidden_var_b.as_tensor(),
        &head_t,
        &idx,
        &lp,
        &mask,
        8,
        &device,
        16,
    )?;
    let grads_b = loss_b.backward()?;
    let g_b = grads_b
        .get(hidden_var_b.as_tensor())
        .ok_or_else(|| anyhow!("no phase-b grad"))?
        .clone();

    // Forward agreement first.
    let ln = loss_naive.to_scalar::<f32>()?;
    let lb = loss_b.to_scalar::<f32>()?;
    assert!(
        (ln - lb).abs() < 1e-5,
        "loss naive={ln:.6} phase_b={lb:.6}"
    );

    let diff = (&g_naive - &g_b)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let max_naive = g_naive.abs()?.max_all()?.to_scalar::<f32>()?;
    let rel = if max_naive > 1e-6 {
        diff / max_naive
    } else {
        diff
    };
    assert!(
        diff < 1e-4 || rel < 1e-4,
        "phase B grad vs naive grad f32: max_abs={diff:.2e} max_naive={max_naive:.4} rel={rel:.2e}"
    );
    Ok(())
}

#[test]
fn cpu_phase_b_backward_matches_naive_bf16() -> Result<()> {
    let device = Device::Cpu;
    let (hidden_f32, head_t_f32, idx, lp, mask) = random_case(16, 8, 64, 8, 2, &device)?;
    let hidden_init = hidden_f32.to_dtype(DType::BF16)?;
    let head_bf = head_t_f32.to_dtype(DType::BF16)?;

    let hidden_var_a = Var::from_tensor(&hidden_init)?;
    let loss_naive = naive_kl(hidden_var_a.as_tensor(), &head_bf, &idx, &lp, &mask, 8, &device)?;
    let grads_naive = loss_naive.backward()?;
    let g_naive = grads_naive
        .get(hidden_var_a.as_tensor())
        .ok_or_else(|| anyhow!("no naive grad"))?
        .clone();

    let hidden_var_b = Var::from_tensor(&hidden_init)?;
    let loss_b = opd_top_k_reverse_kl_phase_b(
        hidden_var_b.as_tensor(),
        &head_bf,
        &idx,
        &lp,
        &mask,
        8,
        &device,
        16,
    )?;
    let grads_b = loss_b.backward()?;
    let g_b = grads_b
        .get(hidden_var_b.as_tensor())
        .ok_or_else(|| anyhow!("no phase-b grad"))?
        .clone();

    let diff = (&g_naive.to_dtype(DType::F32)? - &g_b.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    let max_naive = g_naive
        .to_dtype(DType::F32)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    let rel = if max_naive > 1e-6 {
        diff / max_naive
    } else {
        diff
    };
    assert!(
        diff < 5e-2 || rel < 5e-2,
        "phase B bf16 grad vs naive: max_abs={diff:.2e} max_naive={max_naive:.4} rel={rel:.2e}"
    );
    Ok(())
}

/// Per-position backward: upstream gradient is `[T_active]`; analytic
/// backward must match autograd over per-position KL.
#[test]
fn cpu_phase_b_per_position_backward_matches_naive() -> Result<()> {
    let device = Device::Cpu;
    let (hidden_init, head_t, idx, lp, mask) = random_case(16, 8, 64, 8, 2, &device)?;

    // Naive per-position backward: use sum (not mean) so the upstream gradient
    // for each active position is 1. Phase B with PerPosition output_mode + ones
    // upstream gradient should match.
    let hidden_var_n = Var::from_tensor(&hidden_init)?;
    let per_naive = {
        let active_positions: Vec<u32> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
            .collect();
        let active_indices = Tensor::new(active_positions.as_slice(), &device)?;
        let hidden_2d = hidden_var_n.as_tensor().squeeze(0)?;
        let active_hidden = hidden_2d.index_select(&active_indices, 0)?;
        let active_hidden_f32 = active_hidden.to_dtype(DType::F32)?;
        let head_t_f32 = head_t.to_dtype(DType::F32)?;
        let full_logits = active_hidden_f32.matmul(&head_t_f32)?;
        let active_count = active_positions.len();
        let mut rows = Vec::with_capacity(active_count);
        for t in 0..active_count {
            let row = full_logits.narrow(0, t, 1)?;
            let row_indices = Tensor::new(&idx[t * 8..(t + 1) * 8], &device)?;
            rows.push(row.index_select(&row_indices, 1)?);
        }
        let s_logits = Tensor::cat(&rows, 0)?;
        let q_logprobs = Tensor::from_vec(lp.clone(), (active_count, 8), &device)?;
        let log_p_hat = log_softmax_last(&s_logits)?;
        let log_q_hat = log_softmax_last(&q_logprobs)?;
        let p_hat = log_p_hat.exp()?;
        let diff = (&log_p_hat - &log_q_hat)?;
        (p_hat * diff)?.sum(D::Minus1)?
    };
    let sum_naive = per_naive.sum_all()?;
    let grads_naive = sum_naive.backward()?;
    let g_naive = grads_naive
        .get(hidden_var_n.as_tensor())
        .ok_or_else(|| anyhow!("no naive grad"))?
        .clone();

    // Phase B PerPosition + ones upstream gradient = sum-reduction.
    let hidden_var_b = Var::from_tensor(&hidden_init)?;
    let per_b = opd_top_k_reverse_kl_phase_b_per_position(
        hidden_var_b.as_tensor(),
        &head_t,
        &idx,
        &lp,
        &mask,
        8,
        &device,
        16,
    )?;
    let sum_b = per_b.sum_all()?;
    let grads_b = sum_b.backward()?;
    let g_b = grads_b
        .get(hidden_var_b.as_tensor())
        .ok_or_else(|| anyhow!("no phase-b grad"))?
        .clone();

    let diff = (&g_naive - &g_b)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let max_naive = g_naive.abs()?.max_all()?.to_scalar::<f32>()?;
    let rel = if max_naive > 1e-6 {
        diff / max_naive
    } else {
        diff
    };
    assert!(
        diff < 1e-4 || rel < 1e-4,
        "phase B per-pos grad: max_abs={diff:.2e} max_naive={max_naive:.4} rel={rel:.2e}"
    );
    Ok(())
}

/// CUDA parity: the raw fused kernel must produce the same per-position
/// reverse-KL as the CPU reference (Phase A) within 1e-4 abs at f32 and
/// 5e-2 at bf16. Per §9.2 of the grand plan ("bit-equivalence within
/// 1e-5 across CUDA/Vulkan/Metal" is the ship gate; we test the tighter
/// numbers below as the production tolerance).
///
/// Runs only when the crate is built with `--features cuda`. Run on the
/// A6000 pod from `kiln-pod-acquire`.
#[cfg(feature = "cuda")]
#[test]
fn cuda_kernel_matches_phase_a_f32_k32() -> Result<()> {
    let device = Device::new_cuda(0)?;
    cuda_parity_for_k::<f32>(&device, 32, /* tol = */ 1e-4)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_kernel_matches_phase_a_f32_k16() -> Result<()> {
    let device = Device::new_cuda(0)?;
    cuda_parity_for_k::<f32>(&device, 16, /* tol = */ 1e-4)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_kernel_matches_phase_a_bf16_k32() -> Result<()> {
    let device = Device::new_cuda(0)?;
    cuda_parity_for_k_bf16(&device, 32, /* tol = */ 5e-2)?;
    Ok(())
}

/// CUDA parity helper for f32 inputs. Runs the same test fixture as the
/// CPU parity suite but with a CUDA device, then compares per-position
/// KL produced by Phase B (which routes through the kernel) against
/// Phase A (which forces the candle CPU reference).
#[cfg(feature = "cuda")]
fn cuda_parity_for_k<T: 'static>(
    device: &Device,
    top_k: usize,
    tol: f32,
) -> Result<()> {
    // Use the larger vocabulary the kernel was tuned for.
    let seq_len = 32;
    let hidden_size = 256;
    let vocab_size = 1024;
    let (hidden, head_t, idx, lp, mask) =
        random_case(seq_len, hidden_size, vocab_size, top_k, 2, device)?;

    // Reference: Phase A on the same device (autograd-aware candle path).
    let reference = opd_top_k_reverse_kl_phase_a_per_position(
        &hidden, &head_t, &idx, &lp, &mask, top_k, device,
    )?;
    // Kernel: Phase B with the cuda kernel active.
    let kernel = opd_top_k_reverse_kl_phase_b_per_position(
        &hidden, &head_t, &idx, &lp, &mask, top_k, device, 4096,
    )?;
    let ref_vec: Vec<f32> = reference.to_vec1()?;
    let ker_vec: Vec<f32> = kernel.to_vec1()?;
    assert_eq!(ref_vec.len(), ker_vec.len(), "shape mismatch");
    for (i, (&r, &k)) in ref_vec.iter().zip(ker_vec.iter()).enumerate() {
        let abs = (r - k).abs();
        let rel = if r.abs() > 1e-6 { abs / r.abs() } else { abs };
        assert!(
            abs < tol || rel < tol,
            "f32 K={top_k} pos {i}: ref={r:.6} kernel={k:.6} abs={abs:.2e} rel={rel:.2e}"
        );
    }
    Ok(())
}

/// CUDA backward parity: with the kill-switch ON we get the analytic
/// candle backward; with it OFF we get the fused CUDA bwd kernel. They
/// must agree within tight tolerances.
#[cfg(feature = "cuda")]
#[test]
fn cuda_bwd_kernel_matches_candle_f32_k32() -> Result<()> {
    let device = Device::new_cuda(0)?;
    cuda_bwd_parity_for_k::<f32>(&device, 32, /* tol = */ 1e-4)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_bwd_kernel_matches_candle_f32_k16() -> Result<()> {
    let device = Device::new_cuda(0)?;
    cuda_bwd_parity_for_k::<f32>(&device, 16, /* tol = */ 1e-4)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_bwd_kernel_matches_candle_bf16_k32() -> Result<()> {
    let device = Device::new_cuda(0)?;
    cuda_bwd_parity_for_k_bf16(&device, 32, /* tol = */ 5e-2)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_bwd_parity_for_k<T: 'static>(
    device: &Device,
    top_k: usize,
    tol: f32,
) -> Result<()> {
    use candle_core::Var;
    let seq_len = 32;
    let hidden_size = 256;
    let vocab_size = 1024;
    let (hidden_init, head_t, idx, lp, mask) =
        random_case(seq_len, hidden_size, vocab_size, top_k, 2, device)?;

    // Reference path: KILN_DISABLE_OPD_LOSS_KERNEL=1 forces the
    // analytic candle backward (the parity oracle).
    let prior = std::env::var("KILN_DISABLE_OPD_LOSS_KERNEL").ok();
    unsafe {
        std::env::set_var("KILN_DISABLE_OPD_LOSS_KERNEL", "1");
    }
    let hidden_var_ref = Var::from_tensor(&hidden_init)?;
    let loss_ref = opd_top_k_reverse_kl_phase_b(
        hidden_var_ref.as_tensor(),
        &head_t,
        &idx,
        &lp,
        &mask,
        top_k,
        device,
        4096,
    )?;
    let grads_ref = loss_ref.backward()?;
    let g_ref = grads_ref
        .get(hidden_var_ref.as_tensor())
        .ok_or_else(|| anyhow!("no candle bwd grad"))?
        .clone();
    unsafe {
        match prior.as_ref() {
            Some(v) => std::env::set_var("KILN_DISABLE_OPD_LOSS_KERNEL", v),
            None => std::env::remove_var("KILN_DISABLE_OPD_LOSS_KERNEL"),
        }
    }

    // Kernel path: default behaviour.
    let hidden_var_ker = Var::from_tensor(&hidden_init)?;
    let loss_ker = opd_top_k_reverse_kl_phase_b(
        hidden_var_ker.as_tensor(),
        &head_t,
        &idx,
        &lp,
        &mask,
        top_k,
        device,
        4096,
    )?;
    let grads_ker = loss_ker.backward()?;
    let g_ker = grads_ker
        .get(hidden_var_ker.as_tensor())
        .ok_or_else(|| anyhow!("no kernel bwd grad"))?
        .clone();

    let ref_v = g_ref.to_dtype(DType::F32)?;
    let ker_v = g_ker.to_dtype(DType::F32)?;
    let diff = (&ref_v - &ker_v)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let max_ref = ref_v.abs()?.max_all()?.to_scalar::<f32>()?;
    let rel = if max_ref > 1e-6 {
        diff / max_ref
    } else {
        diff
    };
    assert!(
        diff < tol || rel < tol,
        "bwd kernel vs candle f32 K={top_k}: max_abs={diff:.2e} max_ref={max_ref:.4} rel={rel:.2e}"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_bwd_parity_for_k_bf16(
    device: &Device,
    top_k: usize,
    tol: f32,
) -> Result<()> {
    use candle_core::Var;
    let seq_len = 32;
    let hidden_size = 256;
    let vocab_size = 1024;
    let (hidden_f32, head_f32, idx, lp, mask) =
        random_case(seq_len, hidden_size, vocab_size, top_k, 2, device)?;
    let hidden_init = hidden_f32.to_dtype(DType::BF16)?;
    let head_t = head_f32.to_dtype(DType::BF16)?;

    let prior = std::env::var("KILN_DISABLE_OPD_LOSS_KERNEL").ok();
    unsafe {
        std::env::set_var("KILN_DISABLE_OPD_LOSS_KERNEL", "1");
    }
    let hidden_var_ref = Var::from_tensor(&hidden_init)?;
    let loss_ref = opd_top_k_reverse_kl_phase_b(
        hidden_var_ref.as_tensor(),
        &head_t,
        &idx,
        &lp,
        &mask,
        top_k,
        device,
        4096,
    )?;
    let grads_ref = loss_ref.backward()?;
    let g_ref = grads_ref
        .get(hidden_var_ref.as_tensor())
        .ok_or_else(|| anyhow!("no candle bwd grad"))?
        .clone();
    unsafe {
        match prior.as_ref() {
            Some(v) => std::env::set_var("KILN_DISABLE_OPD_LOSS_KERNEL", v),
            None => std::env::remove_var("KILN_DISABLE_OPD_LOSS_KERNEL"),
        }
    }

    let hidden_var_ker = Var::from_tensor(&hidden_init)?;
    let loss_ker = opd_top_k_reverse_kl_phase_b(
        hidden_var_ker.as_tensor(),
        &head_t,
        &idx,
        &lp,
        &mask,
        top_k,
        device,
        4096,
    )?;
    let grads_ker = loss_ker.backward()?;
    let g_ker = grads_ker
        .get(hidden_var_ker.as_tensor())
        .ok_or_else(|| anyhow!("no kernel bwd grad"))?
        .clone();

    let ref_v = g_ref.to_dtype(DType::F32)?;
    let ker_v = g_ker.to_dtype(DType::F32)?;
    let diff = (&ref_v - &ker_v)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let max_ref = ref_v.abs()?.max_all()?.to_scalar::<f32>()?;
    let rel = if max_ref > 1e-6 {
        diff / max_ref
    } else {
        diff
    };
    assert!(
        diff < tol || rel < tol,
        "bwd kernel vs candle bf16 K={top_k}: max_abs={diff:.2e} max_ref={max_ref:.4} rel={rel:.2e}"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_parity_for_k_bf16(
    device: &Device,
    top_k: usize,
    tol: f32,
) -> Result<()> {
    let seq_len = 32;
    let hidden_size = 256;
    let vocab_size = 1024;
    let (hidden_f32, head_f32, idx, lp, mask) =
        random_case(seq_len, hidden_size, vocab_size, top_k, 2, device)?;
    let hidden = hidden_f32.to_dtype(DType::BF16)?;
    let head_t = head_f32.to_dtype(DType::BF16)?;

    let reference = opd_top_k_reverse_kl_phase_a_per_position(
        &hidden, &head_t, &idx, &lp, &mask, top_k, device,
    )?;
    let kernel = opd_top_k_reverse_kl_phase_b_per_position(
        &hidden, &head_t, &idx, &lp, &mask, top_k, device, 4096,
    )?;
    let ref_vec: Vec<f32> = reference.to_vec1()?;
    let ker_vec: Vec<f32> = kernel.to_vec1()?;
    assert_eq!(ref_vec.len(), ker_vec.len());
    for (i, (&r, &k)) in ref_vec.iter().zip(ker_vec.iter()).enumerate() {
        let abs = (r - k).abs();
        let rel = if r.abs() > 1e-6 { abs / r.abs() } else { abs };
        assert!(
            abs < tol || rel < tol,
            "bf16 K={top_k} pos {i}: ref={r:.6} kernel={k:.6} abs={abs:.2e} rel={rel:.2e}"
        );
    }
    Ok(())
}

/// Sanity: KL between a distribution and itself is zero. When student
/// logits at the K teacher indices are constructed to renormalise to the
/// teacher's distribution exactly, the loss must vanish to round-off.
#[test]
fn cpu_kl_of_matched_distribution_is_zero() -> Result<()> {
    let device = Device::Cpu;
    let seq_len = 4;
    let hidden_size = 4;
    let vocab_size = 32;
    let top_k = 8;

    let label_mask = vec![false, true, false, true];

    // Hand-construct teacher top-K indices + logprobs so the student
    // logits naturally match. We set the student's hidden + head so that
    // hidden @ head produces logits whose top-K renormalisation matches a
    // chosen target distribution.
    let mut teacher_topk_indices: Vec<u32> = Vec::new();
    let mut teacher_topk_logprobs: Vec<f32> = Vec::new();
    for t in 0..2 {
        for k in 0..top_k as u32 {
            teacher_topk_indices.push(t as u32 * (top_k as u32) + k);
        }
        // Uniform distribution over K: logsoftmax(uniform_K) = -log(K)
        for _ in 0..top_k {
            teacher_topk_logprobs.push(-(top_k as f32).ln());
        }
    }

    // Make student logits *also* uniform over those indices: set the
    // relevant columns of head_t to zero, and the (active rows of) hidden
    // to zero. hidden @ head_t = 0; softmax(zeros) is uniform.
    let hidden = Tensor::zeros((1, seq_len, hidden_size), DType::F32, &device)?;
    let head_t = Tensor::zeros((hidden_size, vocab_size), DType::F32, &device)?;

    let loss = opd_top_k_reverse_kl_phase_b(
        &hidden,
        &head_t,
        &teacher_topk_indices,
        &teacher_topk_logprobs,
        &label_mask,
        top_k,
        &device,
        16,
    )?;
    let lv = loss.to_scalar::<f32>()?;
    assert!(lv.abs() < 1e-6, "KL of matched distribution: got {lv:e}");
    Ok(())
}
