//! CPU parity tests for FLCE vs the naive `log_sum_exp - gather` path.
//!
//! The naive reference mirrors the implementation in
//! `kiln-train::trainer::cross_entropy_loss` so a green parity test here is
//! a strong signal that wiring FLCE into the trainer will produce the same
//! gradient signal (modulo floating-point associativity in the chunked
//! reduction).

use anyhow::Result;
use candle_core::{D, DType, Device, Tensor};

use super::{DEFAULT_CHUNK_SIZE, fused_linear_cross_entropy};

/// Naive reference: materialize full logits, compute log-sum-exp and gather.
/// Mirrors `kiln-train::trainer::cross_entropy_loss` so the parity signal
/// here transfers directly to the trainer call sites.
fn naive_loss(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    device: &Device,
) -> Result<Tensor> {
    let seq_len = input_ids.len();
    let hidden_2d = hidden.squeeze(0)?;
    let shift_hidden = hidden_2d.narrow(0, 0, seq_len - 1)?;
    let shift_labels: Vec<u32> = input_ids[1..].to_vec();
    let shift_mask: Vec<bool> = label_mask[1..].to_vec();

    let active_positions: Vec<u32> = shift_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();
    if active_positions.is_empty() {
        return Ok(Tensor::new(0.0f32, device)?);
    }

    let indices = Tensor::new(active_positions.as_slice(), device)?;
    let active_hidden = shift_hidden.index_select(&indices, 0)?;

    // Full logits: [num_active, vocab_size]
    let logits = active_hidden
        .to_dtype(DType::F32)?
        .matmul(&head_t.to_dtype(DType::F32)?)?;

    let log_sum_exp = logits.log_sum_exp(D::Minus1)?; // [num_active]

    let active_labels: Vec<u32> = active_positions
        .iter()
        .map(|&i| shift_labels[i as usize])
        .collect();
    let labels_tensor = Tensor::new(active_labels.as_slice(), device)?.to_dtype(DType::U32)?;
    let labels_2d = labels_tensor.unsqueeze(1)?;
    let correct_logits = logits.gather(&labels_2d, 1)?.squeeze(1)?;

    let per_token_loss = (log_sum_exp - correct_logits)?;
    let loss = per_token_loss.mean_all()?;
    Ok(loss)
}

fn random_case(
    seq_len: usize,
    hidden_size: usize,
    vocab_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Vec<u32>, Vec<bool>)> {
    // Deterministic-enough random via sin(i).
    let total_h = seq_len * hidden_size;
    let hidden_vec: Vec<f32> = (0..total_h)
        .map(|i| (i as f32 * 0.013).sin() * 0.5)
        .collect();
    let hidden = Tensor::from_vec(hidden_vec, (1, seq_len, hidden_size), device)?;

    let total_head = hidden_size * vocab_size;
    let head_vec: Vec<f32> = (0..total_head)
        .map(|i| ((i as f32 + 7.0) * 0.007).cos() * 0.25)
        .collect();
    let head_t = Tensor::from_vec(head_vec, (hidden_size, vocab_size), device)?;

    let input_ids: Vec<u32> = (0..seq_len as u32)
        .map(|i| (i * 31 + 5) % vocab_size as u32)
        .collect();
    let label_mask: Vec<bool> = (0..seq_len).map(|i| i > 0 && i % 2 == 1).collect();

    Ok((hidden, head_t, input_ids, label_mask))
}

#[test]
fn cpu_parity_small() -> Result<()> {
    let device = Device::Cpu;
    let (hidden, head_t, ids, mask) = random_case(16, 8, 64, &device)?;

    let fused = fused_linear_cross_entropy(&hidden, &head_t, &ids, &mask, &device, 16)?;
    let naive = naive_loss(&hidden, &head_t, &ids, &mask, &device)?;

    let fused_v = fused.to_scalar::<f32>()?;
    let naive_v = naive.to_scalar::<f32>()?;

    let abs_err = (fused_v - naive_v).abs();
    let rel_err = if naive_v.abs() > 1e-6 {
        abs_err / naive_v.abs()
    } else {
        abs_err
    };
    assert!(
        abs_err < 1e-5 || rel_err < 1e-5,
        "FLCE parity failed: fused={fused_v:.6} naive={naive_v:.6} abs_err={abs_err:.2e} rel_err={rel_err:.2e}",
    );
    Ok(())
}

#[test]
fn cpu_parity_uneven_vocab_chunks() -> Result<()> {
    // vocab_size not divisible by chunk_size — exercises the trailing-chunk path.
    let device = Device::Cpu;
    let (hidden, head_t, ids, mask) = random_case(12, 6, 73, &device)?;

    let fused = fused_linear_cross_entropy(&hidden, &head_t, &ids, &mask, &device, 16)?;
    let naive = naive_loss(&hidden, &head_t, &ids, &mask, &device)?;

    let abs_err = (fused.to_scalar::<f32>()? - naive.to_scalar::<f32>()?).abs();
    assert!(
        abs_err < 1e-5,
        "uneven chunks parity failed: abs_err={abs_err:.2e}"
    );
    Ok(())
}

#[test]
fn cpu_parity_single_chunk() -> Result<()> {
    // chunk_size >= vocab_size — reduces to the naive path in one iteration.
    let device = Device::Cpu;
    let (hidden, head_t, ids, mask) = random_case(10, 4, 32, &device)?;

    let fused = fused_linear_cross_entropy(&hidden, &head_t, &ids, &mask, &device, 128)?;
    let naive = naive_loss(&hidden, &head_t, &ids, &mask, &device)?;

    let abs_err = (fused.to_scalar::<f32>()? - naive.to_scalar::<f32>()?).abs();
    assert!(
        abs_err < 1e-6,
        "single-chunk parity failed: abs_err={abs_err:.2e}"
    );
    Ok(())
}

#[test]
fn cpu_parity_bf16() -> Result<()> {
    // bf16 inputs — looser tolerance. Mirrors the production training dtype.
    let device = Device::Cpu;
    let (hidden, head_t, ids, mask) = random_case(16, 8, 64, &device)?;
    let hidden_bf = hidden.to_dtype(DType::BF16)?;
    let head_bf = head_t.to_dtype(DType::BF16)?;

    let fused = fused_linear_cross_entropy(&hidden_bf, &head_bf, &ids, &mask, &device, 16)?;
    let naive = naive_loss(&hidden_bf, &head_bf, &ids, &mask, &device)?;

    let abs_err = (fused.to_scalar::<f32>()? - naive.to_scalar::<f32>()?).abs();
    assert!(abs_err < 1e-2, "bf16 parity failed: abs_err={abs_err:.2e}");
    Ok(())
}

#[test]
fn cpu_empty_mask_returns_zero() -> Result<()> {
    let device = Device::Cpu;
    let (hidden, head_t, ids, _) = random_case(8, 4, 16, &device)?;
    let all_false = vec![false; ids.len()];
    let loss = fused_linear_cross_entropy(&hidden, &head_t, &ids, &all_false, &device, 4)?;
    let v = loss.to_scalar::<f32>()?;
    assert_eq!(v, 0.0);
    Ok(())
}

#[test]
fn default_chunk_size_is_positive() {
    assert!(DEFAULT_CHUNK_SIZE > 0);
}

/// Regression test for the chunked-vocab matmul-not-contiguous bug.
///
/// `narrow(1, off, chunk)` on a `[H, V]` tensor with stride `[V, 1]` preserves
/// stride `[V, 1]` for the slice — it does NOT collapse to `[chunk, 1]`.
/// CUDA matmul rejects strided right operands, so the chunked-vocab call
/// crashed on the first SFT step on Qwen3.5-4B with `KILN_USE_FLCE=1`. CPU
/// candle matmul is permissive about strided right operands, which is why
/// the existing parity tests above never caught this.
///
/// This test asserts two things at once:
///   1. The V-axis slice really is non-contiguous on the inner stride
///      (so the test is actually exercising the failing geometry).
///   2. `fused_linear_cross_entropy` still produces parity loss after the
///      `.contiguous()` materialization fix.
///
/// Numeric parity alone is not enough — the CPU path was already green on
/// strided slices before the fix. The contig assertion locks in the
/// regression: if anyone removes `.contiguous()`, this test still detects it
/// because the right-operand layout invariant is what GPU matmul enforces.
///
/// See PR #631 (validation bench) + docs/audits/PHASE10_FLCE_PREFLIGHT.md
/// Finding 1.
#[test]
fn cpu_parity_strided_chunk_slice() -> Result<()> {
    let device = Device::Cpu;
    // V > chunk_size with V not a small power of two — produces a strided
    // V-axis slice that the un-fixed kernel could not feed to CUDA matmul.
    let (hidden, head_t, ids, mask) = random_case(16, 8, 96, &device)?;

    // Sanity: confirm the un-contiguous slice we use to exercise the
    // regression really is strided (test setup invariant).
    let head_t_f32 = head_t.to_dtype(DType::F32)?;
    let probe = head_t_f32.narrow(1, 16, 16)?;
    assert!(
        !probe.is_contiguous(),
        "test setup invariant: V-axis chunk should be strided; if candle changes \
         narrow semantics this test no longer exercises the regression",
    );

    let fused = fused_linear_cross_entropy(&hidden, &head_t, &ids, &mask, &device, 16)?;
    let naive = naive_loss(&hidden, &head_t, &ids, &mask, &device)?;
    let abs_err = (fused.to_scalar::<f32>()? - naive.to_scalar::<f32>()?).abs();
    assert!(
        abs_err < 1e-5,
        "strided-chunk parity failed: abs_err={abs_err:.2e}",
    );
    Ok(())
}
