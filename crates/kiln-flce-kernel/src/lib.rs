//! Fused Linear Cross-Entropy (FLCE).
//!
//! Computes `mean_i ( log_sum_exp_j( hidden[i] @ weight[j].T ) - hidden[i] @ weight[target[i]] )`
//! without ever materialising the full `[N, V]` logits tensor.
//!
//! The forward loops over vocab chunks, maintaining an online `(running_max, running_sumexp)`
//! in F32, and caches only a `[N]` F32 LSE vector for backward. The backward pass recomputes
//! chunked logits, applies softmax, and accumulates `grad_hidden` chunk-by-chunk via
//! `softmax_chunk @ weight_chunk`, then subtracts `weight[target]` and scales by
//! `grad_loss / N`.
//!
//! The LM head weight is assumed frozen (Qwen3.5-4B weight-ties the LM head with
//! `embed_tokens`, and the embedding is not a LoRA target in kiln-train). Only
//! `grad_hidden` is returned from backward.
//!
//! Algorithm matches the Liger-Kernel FLCE description in
//! `docs/PHASE10_LIGER_AUDIT.md` and is gated behind `KILN_DISABLE_FLCE` at the
//! trainer call-sites.

use candle_core::{
    CpuStorage, CustomOp1, DType, Error, Layout, Result, Shape, Tensor,
};

/// Default vocab-chunk size. Picked so each chunk's `[N, chunk]` F32 temporary
/// stays well under 2 GiB even at `N = 65536` (max training context in flight).
pub const DEFAULT_CHUNK_SIZE: usize = 4096;

/// Fused linear cross-entropy with **mean** reduction.
///
/// * `hidden`    : `[N, H]` bf16 (autograd-tracked activations *after* final RMSNorm)
/// * `weight`    : `[V, H]` bf16 (LM head, frozen — no gradient is computed for it)
/// * `targets`   : `[N]` u32 (label ids; positions masked out by the caller are already dropped)
/// * `chunk_size`: vocab tile. `0` → `DEFAULT_CHUNK_SIZE`.
///
/// Returns a scalar F32 tensor with backward populating `grad(hidden)`.
pub fn flce_cross_entropy_loss_mean(
    hidden: &Tensor,
    weight: &Tensor,
    targets: &Tensor,
    chunk_size: usize,
) -> Result<Tensor> {
    let (n, h) = hidden.dims2()?;
    let (v, h2) = weight.dims2()?;
    if h != h2 {
        return Err(Error::Msg(format!(
            "flce: hidden dim {h} != weight hidden dim {h2}"
        )));
    }
    let nt = targets.dims1()?;
    if n != nt {
        return Err(Error::Msg(format!(
            "flce: N mismatch — hidden has {n} rows, targets has {nt}"
        )));
    }
    if hidden.dtype() != DType::BF16 {
        return Err(Error::Msg(format!(
            "flce: hidden must be bf16, got {:?}",
            hidden.dtype()
        )));
    }
    if weight.dtype() != DType::BF16 {
        return Err(Error::Msg(format!(
            "flce: weight must be bf16, got {:?}",
            weight.dtype()
        )));
    }
    if targets.dtype() != DType::U32 {
        return Err(Error::Msg(format!(
            "flce: targets must be u32, got {:?}",
            targets.dtype()
        )));
    }

    if n == 0 {
        return Tensor::zeros((), DType::F32, hidden.device());
    }

    let chunk = if chunk_size == 0 {
        DEFAULT_CHUNK_SIZE
    } else {
        chunk_size
    };

    // ------------- FORWARD (no graph retention beyond scalar loss + [N] F32 LSE) -------------
    let hidden_det = hidden.detach();
    let weight_det = weight.detach();

    // target_logits[i] = sum_k hidden_det[i, k] * weight_det[targets[i], k]  in F32.
    // index_select materialises [N, H] bf16 (= 16384*2560*2 = 80 MiB at T=16384). Tolerable.
    let target_logits_f32 = {
        let w_rows = weight_det.index_select(targets, 0)?.to_dtype(DType::F32)?;
        let h_f32 = hidden_det.to_dtype(DType::F32)?;
        (&h_f32 * &w_rows)?.sum(1)? // [N] F32
    };

    // Chunked online softmax over V to get LSE [N] F32.
    let mut running_max: Option<Tensor> = None; // [N, 1] F32
    let mut running_sumexp: Option<Tensor> = None; // [N, 1] F32

    let mut v_start = 0usize;
    while v_start < v {
        let v_end = (v_start + chunk).min(v);
        let chunk_v = v_end - v_start;

        let w_chunk = weight_det.narrow(0, v_start, chunk_v)?.contiguous()?;
        let w_chunk_t = w_chunk.t()?.contiguous()?;
        let logits_f32 = hidden_det.matmul(&w_chunk_t)?.to_dtype(DType::F32)?;
        let chunk_max = logits_f32.max_keepdim(1)?; // [N, 1]

        let new_max = match &running_max {
            None => chunk_max,
            Some(prev) => prev.maximum(&chunk_max)?,
        };

        let chunk_sum = logits_f32
            .broadcast_sub(&new_max)?
            .exp()?
            .sum_keepdim(1)?;

        let new_sumexp = match (&running_max, &running_sumexp) {
            (None, _) => chunk_sum,
            (Some(prev_max), Some(prev_sum)) => {
                let rescale = (prev_max - &new_max)?.exp()?; // [N, 1]
                let rescaled = prev_sum.mul(&rescale)?;
                rescaled.add(&chunk_sum)?
            }
            _ => unreachable!(),
        };

        running_max = Some(new_max);
        running_sumexp = Some(new_sumexp);
        v_start = v_end;
    }

    let max_final = running_max.expect("loop covers at least one chunk");
    let sum_final = running_sumexp.expect("loop covers at least one chunk");
    let lse = (max_final.squeeze(1)? + sum_final.squeeze(1)?.log()?)?; // [N] F32

    let per_sample = (&lse - &target_logits_f32)?;
    let loss_val: f32 = per_sample.mean_all()?.to_scalar::<f32>()?;

    // ------------- WRAP: CustomOp1 carries grad info back to `hidden` -------------
    let op = FlceBwd {
        weight: weight.clone(),
        targets: targets.clone(),
        lse,
        chunk_size: chunk,
        loss_val,
        num_active: n,
    };
    hidden.apply_op1(op)
}

struct FlceBwd {
    weight: Tensor,
    targets: Tensor,
    lse: Tensor, // [N] F32
    chunk_size: usize,
    loss_val: f32,
    num_active: usize,
}

impl CustomOp1 for FlceBwd {
    fn name(&self) -> &'static str {
        "flce_loss_mean"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        // Return the precomputed scalar loss as a 0-d F32 tensor.
        Ok((CpuStorage::F32(vec![self.loss_val]), Shape::from(())))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        _l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc;

        let device = s.device.clone();
        let stream = device.cuda_stream();
        let host = [self.loss_val];
        // Copy single f32 to device.
        let slice: cudarc::driver::CudaSlice<f32> = stream
            .memcpy_stod(&host)
            .map_err(|e| Error::Msg(format!("flce: memcpy_stod scalar loss: {e}")))?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice::<f32>(slice, device),
            Shape::from(()),
        ))
    }

    fn bwd(&self, hidden: &Tensor, _res: &Tensor, grad_loss: &Tensor) -> Result<Option<Tensor>> {
        // grad_loss is scalar F32 in autograd's grad store.
        let grad_scalar: f32 = grad_loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let scale = (grad_scalar / self.num_active as f32) as f64;

        // Detach for the backward computation — we build a tiny throwaway graph here but
        // never differentiate it again.
        let hidden_det = hidden.detach();
        let weight_det = self.weight.detach();
        let lse = self.lse.unsqueeze(1)?; // [N, 1] F32

        let (_n, _h) = hidden_det.dims2()?;
        let (v, _) = weight_det.dims2()?;

        let mut grad_acc: Option<Tensor> = None;

        let chunk = self.chunk_size.max(1);
        let mut v_start = 0usize;
        while v_start < v {
            let v_end = (v_start + chunk).min(v);
            let chunk_v = v_end - v_start;

            let w_chunk = weight_det.narrow(0, v_start, chunk_v)?.contiguous()?;
            let w_chunk_t = w_chunk.t()?.contiguous()?;

            // logits_chunk: [N, chunk] bf16 → F32
            let logits_f32 = hidden_det.matmul(&w_chunk_t)?.to_dtype(DType::F32)?;
            // softmax_chunk = exp(logits - full_LSE)  — broadcast subtract along V dim
            let softmax_chunk = logits_f32.broadcast_sub(&lse)?.exp()?; // [N, chunk] F32

            // Accumulate softmax @ weight_chunk in F32 to avoid bf16 accumulation drift.
            let w_chunk_f32 = w_chunk.to_dtype(DType::F32)?;
            let contrib = softmax_chunk.matmul(&w_chunk_f32)?; // [N, H] F32

            grad_acc = Some(match grad_acc.take() {
                None => contrib,
                Some(prev) => prev.add(&contrib)?,
            });

            v_start = v_end;
        }

        let mut grad = grad_acc.expect("at least one chunk");

        // Subtract weight[targets] (the one-hot component of softmax - one_hot).
        let target_w = weight_det
            .index_select(&self.targets, 0)?
            .to_dtype(DType::F32)?;
        grad = grad.sub(&target_w)?;

        // Scale by grad_loss / N and cast back to bf16.
        let grad = grad.affine(scale, 0.0)?;
        let grad = grad.to_dtype(DType::BF16)?;

        Ok(Some(grad))
    }
}

/// Reference implementation that *does* materialise `[N, V]` logits and then runs the standard
/// `softmax` + NLL path, matching the semantics of `kiln_train::trainer::cross_entropy_loss`
/// after active-row filtering. Useful for parity tests.
pub fn reference_cross_entropy_loss_mean(
    hidden: &Tensor,
    weight: &Tensor,
    targets: &Tensor,
) -> Result<Tensor> {
    let logits = hidden.matmul(&weight.t()?.contiguous()?)?; // [N, V] bf16
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let lse = log_sum_exp(&logits_f32, 1)?; // [N]
    let targets_i = targets.unsqueeze(1)?; // [N, 1]
    let target_logits = logits_f32.gather(&targets_i, 1)?.squeeze(1)?; // [N]
    (lse - target_logits)?.mean_all()
}

fn log_sum_exp(x: &Tensor, dim: usize) -> Result<Tensor> {
    let max = x.max_keepdim(dim)?;
    let shifted = x.broadcast_sub(&max)?;
    let sum = shifted.exp()?.sum_keepdim(dim)?;
    (max + sum.log()?)?.squeeze(dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    fn random_hidden(n: usize, h: usize, device: &Device) -> Result<Tensor> {
        Tensor::randn(0.0f32, 0.02, (n, h), device)?.to_dtype(DType::BF16)
    }

    fn random_weight(v: usize, h: usize, device: &Device) -> Result<Tensor> {
        Tensor::randn(0.0f32, 0.02, (v, h), device)?.to_dtype(DType::BF16)
    }

    fn random_targets(n: usize, v: u32, device: &Device) -> Result<Tensor> {
        let mut rng_state = 0x12345u64;
        let vals: Vec<u32> = (0..n)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rng_state >> 32) as u32) % v
            })
            .collect();
        Tensor::from_vec(vals, (n,), device)
    }

    #[test]
    fn forward_parity_small() -> Result<()> {
        let device = Device::Cpu;
        let n = 17;
        let h = 64;
        let v = 256;

        let hidden = random_hidden(n, h, &device)?;
        let weight = random_weight(v, h, &device)?;
        let targets = random_targets(n, v as u32, &device)?;

        let ref_loss = reference_cross_entropy_loss_mean(&hidden, &weight, &targets)?
            .to_scalar::<f32>()?;
        let flce_loss = flce_cross_entropy_loss_mean(&hidden, &weight, &targets, 32)?
            .to_scalar::<f32>()?;

        assert!(
            (ref_loss - flce_loss).abs() < 1e-3,
            "ref={ref_loss} flce={flce_loss}"
        );
        Ok(())
    }

    #[test]
    fn forward_parity_multiple_chunks() -> Result<()> {
        // Chunk size < V/4 to exercise the online reduction.
        let device = Device::Cpu;
        let n = 33;
        let h = 48;
        let v = 1024;

        let hidden = random_hidden(n, h, &device)?;
        let weight = random_weight(v, h, &device)?;
        let targets = random_targets(n, v as u32, &device)?;

        let ref_loss = reference_cross_entropy_loss_mean(&hidden, &weight, &targets)?
            .to_scalar::<f32>()?;
        // Try several chunk sizes, including unaligned ones.
        for chunk in [32usize, 64, 100, 257, 1024, 2048] {
            let flce_loss = flce_cross_entropy_loss_mean(&hidden, &weight, &targets, chunk)?
                .to_scalar::<f32>()?;
            assert!(
                (ref_loss - flce_loss).abs() < 1e-3,
                "chunk={chunk} ref={ref_loss} flce={flce_loss}"
            );
        }
        Ok(())
    }

    #[test]
    fn backward_parity_grad_hidden() -> Result<()> {
        let device = Device::Cpu;
        let n = 13;
        let h = 32;
        let v = 128;

        // Build a Var over hidden so autograd tracks it. Use F32 master copy, cast to bf16
        // for both paths (so gradients propagate through the cast in the reference case).
        let hidden_master = Var::from_tensor(&Tensor::randn(0.0f32, 0.02, (n, h), &device)?)?;
        let weight = random_weight(v, h, &device)?;
        let targets = random_targets(n, v as u32, &device)?;

        // Reference: cast→matmul→CE, backward through candle.
        let hidden_bf16_ref = hidden_master.as_tensor().to_dtype(DType::BF16)?;
        let ref_loss = reference_cross_entropy_loss_mean(&hidden_bf16_ref, &weight, &targets)?;
        let ref_grads = ref_loss.backward()?;
        let ref_grad = ref_grads
            .get(hidden_master.as_tensor())
            .expect("ref grad for hidden")
            .clone();

        // FLCE: cast→flce, backward.
        let hidden_bf16_flce = hidden_master.as_tensor().to_dtype(DType::BF16)?;
        let flce_loss = flce_cross_entropy_loss_mean(&hidden_bf16_flce, &weight, &targets, 32)?;
        let flce_grads = flce_loss.backward()?;
        let flce_grad = flce_grads
            .get(hidden_master.as_tensor())
            .expect("flce grad for hidden")
            .clone();

        let diff = (&ref_grad - &flce_grad)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-2, "grad parity: max abs diff = {diff}");
        Ok(())
    }

    #[test]
    fn empty_input_returns_zero() -> Result<()> {
        let device = Device::Cpu;
        let hidden = random_hidden(0, 16, &device)?;
        let weight = random_weight(32, 16, &device)?;
        let targets = Tensor::from_vec::<u32>(vec![], (0,), &device)?;

        let loss = flce_cross_entropy_loss_mean(&hidden, &weight, &targets, 8)?;
        assert_eq!(loss.to_scalar::<f32>()?, 0.0);
        Ok(())
    }
}
