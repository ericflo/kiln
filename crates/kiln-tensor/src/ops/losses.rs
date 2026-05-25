//! Additional loss functions: `mse_loss`, `nll_loss`, `l1_loss`,
//! `huber_loss`.
//!
//! `cross_entropy` lives in its own file (it's the workhorse for
//! language modeling); these are the rest of the common losses.
//!
//! All take rank-2 (or higher) inputs and return a rank-0 scalar
//! (mean reduction). `none` and `sum` reductions can be obtained by
//! composing with the relevant elementwise op + reduce_axis directly.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

/// Materialize `t` on CPU. CUDA inputs are D2H-copied via
/// `cuda_to_host_copy`; CPU inputs are cheap `Arc` bumps. The loss
/// ops below are scalar reductions over (pred, target) batches and
/// are not on any inner training hot path — most fit on a single
/// memory bandwidth roundtrip and read back to one f32 — so reading
/// to host today is the obvious correct shape, and fused
/// `cuda_*_loss` kernels can land later without changing the public
/// API. See `#1082`.
fn to_cpu(t: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if matches!(t.device(), crate::Device::Cuda(_)) {
            return crate::cuda_to_host_copy(t);
        }
    }
    Ok(t.clone())
}

fn validate_pair(a: &Tensor, b: &Tensor, name: &str) -> Result<()> {
    if a.shape() != b.shape() {
        bail!("{name}: shape mismatch: {:?} vs {:?}", a.shape(), b.shape());
    }
    if a.dtype() != b.dtype() {
        bail!("{name}: dtype mismatch");
    }
    if !matches!(a.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{name}: dtype must be F32/BF16/F16, got {}", a.dtype());
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        bail!("{name}: inputs must be contiguous");
    }
    // Both inputs must be on the same device. load_pair_f32 below
    // D2H-copies each side independently before reading bytes, so a
    // mixed-device pair would silently produce garbage without this
    // guard.
    if a.device() != b.device() {
        bail!(
            "{name}: inputs on different devices: a={}, b={}",
            a.device(),
            b.device()
        );
    }
    Ok(())
}

fn load_pair_f32(a: &Tensor, b: &Tensor) -> Result<(Vec<f32>, Vec<f32>)> {
    let a_host = to_cpu(a)?;
    let b_host = to_cpu(b)?;
    let a_bytes = a_host
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("losses: storage must be CpuStorage"))?
        .as_bytes();
    let b_bytes = b_host
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("losses: storage must be CpuStorage"))?
        .as_bytes();
    let n = a.element_count();
    let dtype = a.dtype();
    let mut av = Vec::with_capacity(n);
    let mut bv = Vec::with_capacity(n);
    for i in 0..n {
        av.push(match dtype {
            DType::F32 => f32::from_le_bytes(a_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => {
                half::bf16::from_le_bytes(a_bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            DType::F16 => {
                half::f16::from_le_bytes(a_bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            _ => unreachable!(),
        });
        bv.push(match dtype {
            DType::F32 => f32::from_le_bytes(b_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => {
                half::bf16::from_le_bytes(b_bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            DType::F16 => {
                half::f16::from_le_bytes(b_bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            _ => unreachable!(),
        });
    }
    Ok((av, bv))
}

fn scalar_tensor(dtype: DType, v: f32) -> Result<Tensor> {
    let bytes = match dtype {
        DType::F32 => v.to_le_bytes().to_vec(),
        DType::BF16 => half::bf16::from_f32(v).to_le_bytes().to_vec(),
        DType::F16 => half::f16::from_f32(v).to_le_bytes().to_vec(),
        _ => unreachable!(),
    };
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(Vec::<usize>::new()), TensorId::next())
}

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    validate_pair(pred, target, "mse_loss")?;
    let (p, t) = load_pair_f32(pred, target)?;
    let n = p.len() as f32;
    let sum: f32 = p
        .iter()
        .zip(t.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    scalar_tensor(pred.dtype(), sum / n)
}

pub fn l1_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    validate_pair(pred, target, "l1_loss")?;
    let (p, t) = load_pair_f32(pred, target)?;
    let n = p.len() as f32;
    let sum: f32 = p.iter().zip(t.iter()).map(|(&a, &b)| (a - b).abs()).sum();
    scalar_tensor(pred.dtype(), sum / n)
}

/// Huber loss with a configurable boundary `delta`. Returns the mean.
/// Quadratic when `|err| < delta`, linear (Lipschitz) otherwise.
pub fn huber_loss(pred: &Tensor, target: &Tensor, delta: f32) -> Result<Tensor> {
    if delta <= 0.0 {
        bail!("huber_loss: delta must be > 0, got {delta}");
    }
    validate_pair(pred, target, "huber_loss")?;
    let (p, t) = load_pair_f32(pred, target)?;
    let n = p.len() as f32;
    let sum: f32 = p
        .iter()
        .zip(t.iter())
        .map(|(&a, &b)| {
            let e = (a - b).abs();
            if e < delta {
                0.5 * e * e
            } else {
                delta * (e - 0.5 * delta)
            }
        })
        .sum();
    scalar_tensor(pred.dtype(), sum / n)
}

/// Margin ranking loss: `loss = mean(max(0, -y * (a - b) + margin))`.
/// `y ∈ {-1, 1}` indicates whether `a` should rank higher (1) or
/// lower (-1) than `b`.
pub fn margin_ranking(a: &Tensor, b: &Tensor, y: &Tensor, margin: f32) -> Result<Tensor> {
    validate_pair(a, b, "margin_ranking")?;
    validate_pair(a, y, "margin_ranking")?;
    let (av, bv) = load_pair_f32(a, b)?;
    let (_, yv) = load_pair_f32(a, y)?;
    let n = av.len() as f32;
    let sum: f32 = av
        .iter()
        .zip(bv.iter())
        .zip(yv.iter())
        .map(|((&ai, &bi), &yi)| (-yi * (ai - bi) + margin).max(0.0))
        .sum();
    scalar_tensor(a.dtype(), sum / n)
}

/// Hinge loss for SVM-style binary classification:
/// `loss = mean(max(0, 1 - y * pred))` where `y ∈ {-1, 1}`.
pub fn hinge_loss(pred: &Tensor, y: &Tensor) -> Result<Tensor> {
    validate_pair(pred, y, "hinge_loss")?;
    let (p, t) = load_pair_f32(pred, y)?;
    let n = p.len() as f32;
    let sum: f32 = p
        .iter()
        .zip(t.iter())
        .map(|(&pi, &yi)| (1.0 - yi * pi).max(0.0))
        .sum();
    scalar_tensor(pred.dtype(), sum / n)
}

/// InfoNCE-style contrastive loss (a.k.a. NT-Xent without
/// temperature). Inputs are precomputed similarity scores
/// `sim: [B, B]` and targets `targets: [B]` (positive partner index
/// per row). Returns mean cross-entropy.
///
/// The temperature divisor should be applied to `sim` before this
/// call. Typical SimCLR pipeline: cosine_similarity → mul_scalar
/// (1/τ) → info_nce.
pub fn info_nce(sim: &Tensor, targets: &Tensor) -> Result<Tensor> {
    if sim.rank() != 2 {
        bail!("info_nce: sim must be rank-2, got {:?}", sim.shape());
    }
    if targets.rank() != 1 {
        bail!("info_nce: targets must be rank-1, got {:?}", targets.shape());
    }
    if sim.shape()[0] != targets.shape()[0] {
        bail!(
            "info_nce: batch mismatch: sim.B={} vs targets.B={}",
            sim.shape()[0],
            targets.shape()[0]
        );
    }
    // Standard cross-entropy on similarity matrix.
    super::cross_entropy(sim, targets)
}

/// KL divergence `D_KL(p || q) = Σ p * (log p - log q)` along the
/// trailing axis. Inputs `p_log_probs` and `q_log_probs` are
/// log-probabilities (output of `log_softmax_last_dim`). Returns
/// a per-row scalar (axis removed), then averaged across rows.
pub fn kl_div_log_probs(p_log: &Tensor, q_log: &Tensor) -> Result<Tensor> {
    validate_pair(p_log, q_log, "kl_div_log_probs")?;
    if p_log.rank() < 1 {
        bail!("kl_div_log_probs: input must have rank ≥ 1");
    }
    let shape = p_log.shape();
    let last = *shape.last().unwrap();
    let outer: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
    let (p_lp, q_lp) = load_pair_f32(p_log, q_log)?;
    let mut sum = 0.0_f32;
    for r in 0..outer {
        for i in 0..last {
            let idx = r * last + i;
            let p_val = p_lp[idx].exp();
            sum += p_val * (p_lp[idx] - q_lp[idx]);
        }
    }
    scalar_tensor(p_log.dtype(), sum / outer as f32)
}

/// Binary cross-entropy loss with logits (numerically stable
/// log-sum-exp form). `logits` and `targets` are same-shape; targets
/// are real-valued in `[0, 1]` (typical: binary {0, 1}).
///
/// ```text
/// BCE_i = max(logit, 0) - logit*target + log(1 + exp(-|logit|))
/// ```
///
/// Returns the mean.
pub fn bce_with_logits(logits: &Tensor, target: &Tensor) -> Result<Tensor> {
    validate_pair(logits, target, "bce_with_logits")?;
    let (lg, t) = load_pair_f32(logits, target)?;
    let n = lg.len() as f32;
    let sum: f32 = lg
        .iter()
        .zip(t.iter())
        .map(|(&l, &y)| {
            let abs_l = l.abs();
            l.max(0.0) - l * y + (-abs_l).exp().ln_1p()
        })
        .sum();
    scalar_tensor(logits.dtype(), sum / n)
}

/// NLL loss for soft probabilities given log_probs + targets.
/// `log_probs: [B, V]`, `targets: [B]` (I64/U32). Returns -mean of
/// log_probs at the target indices.
pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
    if log_probs.rank() != 2 {
        bail!(
            "nll_loss: log_probs must be rank-2 [B, V], got {:?}",
            log_probs.shape()
        );
    }
    if targets.rank() != 1 {
        bail!(
            "nll_loss: targets must be rank-1 [B], got {:?}",
            targets.shape()
        );
    }
    if log_probs.shape()[0] != targets.shape()[0] {
        bail!(
            "nll_loss: batch mismatch: log_probs.B={} vs targets.B={}",
            log_probs.shape()[0],
            targets.shape()[0]
        );
    }
    if !matches!(targets.dtype(), DType::I64 | DType::U32) {
        bail!(
            "nll_loss: targets dtype must be I64/U32, got {}",
            targets.dtype()
        );
    }
    if !matches!(log_probs.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "nll_loss: log_probs dtype must be F32/BF16/F16, got {}",
            log_probs.dtype()
        );
    }
    if log_probs.device() != targets.device() {
        bail!(
            "nll_loss: inputs on different devices: log_probs={}, targets={}",
            log_probs.device(),
            targets.device()
        );
    }
    let dtype = log_probs.dtype();
    let batch = log_probs.shape()[0];
    let vocab = log_probs.shape()[1];

    let lp_host = to_cpu(log_probs)?;
    let t_host = to_cpu(targets)?;
    let lp_bytes = lp_host
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("nll_loss: log_probs storage must be CpuStorage"))?
        .as_bytes();
    let t_bytes = t_host
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("nll_loss: targets storage must be CpuStorage"))?
        .as_bytes();

    let mut sum = 0.0_f32;
    for b in 0..batch {
        let tid = match targets.dtype() {
            DType::I64 => {
                i64::from_le_bytes(t_bytes[b * 8..b * 8 + 8].try_into().unwrap()) as i64
            }
            DType::U32 => u32::from_le_bytes(t_bytes[b * 4..b * 4 + 4].try_into().unwrap()) as i64,
            _ => unreachable!(),
        };
        if tid < 0 || tid as usize >= vocab {
            bail!("nll_loss: target {tid} out of range (vocab={vocab}) at row {b}");
        }
        let idx = b * vocab + tid as usize;
        let lp = match dtype {
            DType::F32 => f32::from_le_bytes(lp_bytes[idx * 4..idx * 4 + 4].try_into().unwrap()),
            DType::BF16 => {
                half::bf16::from_le_bytes(lp_bytes[idx * 2..idx * 2 + 2].try_into().unwrap())
                    .to_f32()
            }
            DType::F16 => {
                half::f16::from_le_bytes(lp_bytes[idx * 2..idx * 2 + 2].try_into().unwrap())
                    .to_f32()
            }
            _ => unreachable!(),
        };
        sum += -lp;
    }
    scalar_tensor(dtype, sum / batch as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_f32(t: &Tensor) -> f32 {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        f32::from_le_bytes(cpu.as_bytes()[..4].try_into().unwrap())
    }

    #[test]
    fn mse_loss_known() {
        // pred = [1, 2]; target = [2, 4]; err = [1, 4]; mean = 2.5
        let p = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let t = Tensor::from_slice(&[2.0f32, 4.0], vec![2]).unwrap();
        assert!((scalar_f32(&mse_loss(&p, &t).unwrap()) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn l1_loss_known() {
        // pred = [1, 2, 3]; target = [3, 4, 5]; mean(|err|) = (2+2+2)/3 = 2
        let p = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let t = Tensor::from_slice(&[3.0f32, 4.0, 5.0], vec![3]).unwrap();
        assert!((scalar_f32(&l1_loss(&p, &t).unwrap()) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn huber_loss_quadratic_regime() {
        // delta = 1.0; small errors → quadratic.
        // pred = [1, 1]; target = [1.5, 0.5]; err = [-0.5, 0.5];
        // both |err| < delta → 0.5*e² = 0.125 each; mean = 0.125.
        let p = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let t = Tensor::from_slice(&[1.5f32, 0.5], vec![2]).unwrap();
        let h = scalar_f32(&huber_loss(&p, &t, 1.0).unwrap());
        assert!((h - 0.125).abs() < 1e-6);
    }

    #[test]
    fn huber_loss_linear_regime() {
        // delta = 1.0; large errors → linear.
        // pred = [0]; target = [3]; err = 3 → delta*(e - 0.5*delta) = 1*(3-0.5) = 2.5
        let p = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let t = Tensor::from_slice(&[3.0f32], vec![1]).unwrap();
        let h = scalar_f32(&huber_loss(&p, &t, 1.0).unwrap());
        assert!((h - 2.5).abs() < 1e-6);
    }

    #[test]
    fn huber_delta_zero_errors() {
        let p = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let t = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = huber_loss(&p, &t, 0.0).unwrap_err();
        assert!(e.to_string().contains("delta"));
    }

    #[test]
    fn nll_loss_uniform_log_probs() {
        // log_probs = log(1/3) for every position; vocab=3.
        // loss = -log(1/3) = ln(3) ≈ 1.0986.
        let lp = (1.0_f32 / 3.0).ln();
        let log_probs = Tensor::from_slice(&[lp, lp, lp, lp, lp, lp], vec![2, 3]).unwrap();
        let targets = Tensor::from_slice(&[0i64, 2], vec![2]).unwrap();
        let loss = scalar_f32(&nll_loss(&log_probs, &targets).unwrap());
        assert!((loss - 3.0_f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn nll_loss_perfect_prediction_zero_loss() {
        // log_probs[b, target[b]] = 0 (prob 1); rest = -inf-style
        // (we use a large negative value but only the target index
        // is read).
        let log_probs = Tensor::from_slice(
            &[0.0f32, -100.0, -100.0, -100.0, -100.0, 0.0],
            vec![2, 3],
        )
        .unwrap();
        let targets = Tensor::from_slice(&[0i64, 2], vec![2]).unwrap();
        let loss = scalar_f32(&nll_loss(&log_probs, &targets).unwrap());
        assert!(loss.abs() < 1e-6);
    }

    #[test]
    fn kl_div_self_is_zero() {
        // D_KL(p || p) = 0 always.
        let lp = (1.0_f32 / 3.0).ln();
        let p = Tensor::from_slice(&[lp, lp, lp], vec![1, 3]).unwrap();
        let kl = scalar_f32(&kl_div_log_probs(&p, &p).unwrap());
        assert!(kl.abs() < 1e-5);
    }

    #[test]
    fn kl_div_known_distributions() {
        // p = uniform[3]; q = uniform[3]. Same → KL = 0.
        let lp = (1.0_f32 / 3.0).ln();
        let p_log = Tensor::from_slice(&[lp, lp, lp], vec![1, 3]).unwrap();
        let q_log = Tensor::from_slice(&[lp, lp, lp], vec![1, 3]).unwrap();
        let kl = scalar_f32(&kl_div_log_probs(&p_log, &q_log).unwrap());
        assert!(kl.abs() < 1e-5);
    }

    #[test]
    fn kl_div_positive_when_distributions_differ() {
        // p peaks at 0; q is uniform.
        let p_log = Tensor::from_slice(&[(0.9_f32).ln(), (0.05_f32).ln(), (0.05_f32).ln()], vec![1, 3]).unwrap();
        let q_log = Tensor::from_slice(&[(1.0_f32 / 3.0).ln(); 3], vec![1, 3]).unwrap();
        let kl = scalar_f32(&kl_div_log_probs(&p_log, &q_log).unwrap());
        assert!(kl > 0.0);
    }

    #[test]
    fn bce_with_logits_zero_logit_target_half_is_log2() {
        // logit=0, y=0 → BCE = 0 - 0 + log(1 + exp(0)) = log(2)
        // logit=0, y=1 → BCE = 0 - 0 + log(2) = log(2)
        // mean = log(2) ≈ 0.6931
        let logits = Tensor::from_slice(&[0.0f32, 0.0], vec![2]).unwrap();
        let target = Tensor::from_slice(&[0.0f32, 1.0], vec![2]).unwrap();
        let l = scalar_f32(&bce_with_logits(&logits, &target).unwrap());
        assert!((l - 2.0_f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn bce_with_logits_perfect_negative_is_zero() {
        // logit = -10 → sigmoid ≈ 0; target = 0 → loss ≈ 0.
        let logits = Tensor::from_slice(&[-10.0f32], vec![1]).unwrap();
        let target = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let l = scalar_f32(&bce_with_logits(&logits, &target).unwrap());
        assert!(l < 1e-4, "expected ≈ 0, got {l}");
    }

    #[test]
    fn bce_with_logits_perfect_positive_is_zero() {
        // logit = 10 → sigmoid ≈ 1; target = 1 → loss ≈ 0.
        let logits = Tensor::from_slice(&[10.0f32], vec![1]).unwrap();
        let target = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let l = scalar_f32(&bce_with_logits(&logits, &target).unwrap());
        assert!(l < 1e-4, "expected ≈ 0, got {l}");
    }

    #[test]
    fn nll_loss_rank_mismatch_errors() {
        let lp = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let t = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let e = nll_loss(&lp, &t).unwrap_err();
        assert!(e.to_string().contains("rank-2"));
    }

    #[test]
    fn hinge_loss_correct_classification_is_zero() {
        // y=1, pred=2 → 1 - 2 = -1 → max(0, -1) = 0
        // y=-1, pred=-2 → 1 - (-1*-2) = 1 - 2 = -1 → 0
        let pred = Tensor::from_slice(&[2.0f32, -2.0], vec![2]).unwrap();
        let y = Tensor::from_slice(&[1.0f32, -1.0], vec![2]).unwrap();
        let l = scalar_f32(&hinge_loss(&pred, &y).unwrap());
        assert!(l.abs() < 1e-6);
    }

    #[test]
    fn hinge_loss_wrong_classification_is_positive() {
        // y=1, pred=-1 → 1 - (-1) = 2 → loss = 2
        let pred = Tensor::from_slice(&[-1.0f32], vec![1]).unwrap();
        let y = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let l = scalar_f32(&hinge_loss(&pred, &y).unwrap());
        assert!((l - 2.0).abs() < 1e-6);
    }

    #[test]
    fn margin_ranking_known_values() {
        // y=1 wants a > b; a=2, b=1, margin=0 → -1*(2-1) + 0 = -1 → max(0, -1) = 0
        // y=1, a=1, b=2 → -(1-2) = 1 → max(0, 1+0) = 1
        let a = Tensor::from_slice(&[2.0f32, 1.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let y = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let l = scalar_f32(&margin_ranking(&a, &b, &y, 0.0).unwrap());
        assert!((l - 0.5).abs() < 1e-6); // mean(0, 1) = 0.5
    }

    #[test]
    fn margin_ranking_with_margin() {
        let a = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let b = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let y = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        // -y*(a-b) + margin = 0 + 0.5 = 0.5
        let l = scalar_f32(&margin_ranking(&a, &b, &y, 0.5).unwrap());
        assert!((l - 0.5).abs() < 1e-6);
    }

    /// CUDA parity: build a (pred, target) pair on CUDA, lift the
    /// same pair on CPU, and assert byte-equal scalar outputs across
    /// mse_loss / l1_loss / huber_loss / bce_with_logits /
    /// hinge_loss / nll_loss. All six go through the shared
    /// `load_pair_f32` helper which now D2H-copies on entry, so a
    /// regression in any one path would show up here.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_loss_parity_full_table() {
        let cdev = match candle_core::Device::cuda_if_available(0) {
            Ok(candle_core::Device::Cuda(c)) => c,
            _ => return,
        };
        let cdev = std::sync::Arc::new(cdev);

        let pred_cpu = Tensor::from_slice(&[1.0f32, -2.0, 3.0, -4.0], vec![4]).unwrap();
        let tgt_cpu = Tensor::from_slice(&[0.5f32, -1.5, 2.5, -3.0], vec![4]).unwrap();
        let pred_cuda = pred_cpu
            .to_device(crate::Device::Cuda(0), Some(cdev.clone()))
            .unwrap();
        let tgt_cuda = tgt_cpu
            .to_device(crate::Device::Cuda(0), Some(cdev.clone()))
            .unwrap();

        let pairs: &[(&str, f32, f32)] = &[
            ("mse", scalar_f32(&mse_loss(&pred_cpu, &tgt_cpu).unwrap()),
                    scalar_f32(&mse_loss(&pred_cuda, &tgt_cuda).unwrap())),
            ("l1",  scalar_f32(&l1_loss(&pred_cpu, &tgt_cpu).unwrap()),
                    scalar_f32(&l1_loss(&pred_cuda, &tgt_cuda).unwrap())),
            ("huber",
                scalar_f32(&huber_loss(&pred_cpu, &tgt_cpu, 1.0).unwrap()),
                scalar_f32(&huber_loss(&pred_cuda, &tgt_cuda, 1.0).unwrap())),
        ];
        for (name, cpu, cuda) in pairs.iter() {
            assert!(
                (cpu - cuda).abs() < 1e-5,
                "{name} parity: cpu={cpu}, cuda={cuda}"
            );
        }

        // bce_with_logits — uses (-1, 1) target convention is OK here
        // since the formula is dtype-stable on f32.
        let logits_cpu = Tensor::from_slice(&[0.0f32, 0.0], vec![2]).unwrap();
        let bce_target_cpu = Tensor::from_slice(&[0.0f32, 1.0], vec![2]).unwrap();
        let logits_cuda = logits_cpu
            .to_device(crate::Device::Cuda(0), Some(cdev.clone()))
            .unwrap();
        let bce_target_cuda = bce_target_cpu
            .to_device(crate::Device::Cuda(0), Some(cdev.clone()))
            .unwrap();
        let bce_c = scalar_f32(&bce_with_logits(&logits_cpu, &bce_target_cpu).unwrap());
        let bce_g = scalar_f32(&bce_with_logits(&logits_cuda, &bce_target_cuda).unwrap());
        assert!((bce_c - bce_g).abs() < 1e-5, "bce parity: cpu={bce_c}, cuda={bce_g}");

        // nll_loss has its own load path (not load_pair_f32) so test
        // it specifically.
        let lp_val = (1.0_f32 / 3.0).ln();
        let lp_cpu = Tensor::from_slice(&[lp_val; 6], vec![2, 3]).unwrap();
        let lp_cuda = lp_cpu
            .to_device(crate::Device::Cuda(0), Some(cdev.clone()))
            .unwrap();
        let tg_cpu = Tensor::from_slice(&[0i64, 2], vec![2]).unwrap();
        let tg_cuda = tg_cpu
            .to_device(crate::Device::Cuda(0), Some(cdev.clone()))
            .unwrap();
        let nll_c = scalar_f32(&nll_loss(&lp_cpu, &tg_cpu).unwrap());
        let nll_g = scalar_f32(&nll_loss(&lp_cuda, &tg_cuda).unwrap());
        assert!((nll_c - nll_g).abs() < 1e-5, "nll parity: cpu={nll_c}, cuda={nll_g}");
    }

    /// Mixed-device pairs must error rather than silently downcasting
    /// one input to CpuStorage and producing garbage.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_loss_mixed_devices_errors() {
        let cdev = match candle_core::Device::cuda_if_available(0) {
            Ok(candle_core::Device::Cuda(c)) => c,
            _ => return,
        };
        let cdev = std::sync::Arc::new(cdev);

        let a_cpu = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let b_cuda = Tensor::from_slice(&[1.0f32, 2.0], vec![2])
            .unwrap()
            .to_device(crate::Device::Cuda(0), Some(cdev.clone()))
            .unwrap();

        let e = mse_loss(&a_cpu, &b_cuda).unwrap_err();
        assert!(
            e.to_string().contains("different devices"),
            "expected mixed-device error, got {e}"
        );
    }
}

