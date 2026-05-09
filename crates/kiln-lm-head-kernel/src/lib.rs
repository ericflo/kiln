//! CUDA kernels for batched greedy LM-head argmax.
//!
//! The production decode path needs one token id per batch row without
//! materializing `[batch, 1, vocab]` logits. This crate computes BF16
//! `x @ weight_t`, keeps the rowwise argmax reduction on GPU, and copies back
//! only `[batch]` token ids.

use candle_core::{DType, Device, Result, Tensor, cuda_backend::cudarc::driver::DevicePtr};
use half::bf16;

const MAX_BATCH: usize = 32;

unsafe extern "C" {
    fn kiln_lm_head_argmax_bf16_batch(
        x: *const core::ffi::c_void,
        weight_t: *const core::ffi::c_void,
        scores: *mut f32,
        tokens: *mut u32,
        batch: i32,
        hidden: i32,
        vocab: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

pub fn supports_bf16_batch(x: &Tensor, weight_t: &Tensor) -> bool {
    if x.dtype() != DType::BF16 || weight_t.dtype() != DType::BF16 {
        return false;
    }
    if !matches!(x.device(), Device::Cuda(_)) || !matches!(weight_t.device(), Device::Cuda(_)) {
        return false;
    }
    if !x.is_contiguous() || !weight_t.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((weight_hidden, vocab)) = weight_t.dims2() else {
        return false;
    };
    batch > 0
        && batch <= MAX_BATCH
        && seq_len == 1
        && hidden == weight_hidden
        && hidden <= i32::MAX as usize
        && vocab > 0
        && vocab <= i32::MAX as usize
        && batch.checked_mul(vocab).is_some()
}

pub fn argmax_bf16_batch(x: &Tensor, weight_t: &Tensor) -> Result<Vec<u32>> {
    if !supports_bf16_batch(x, weight_t) {
        candle_core::bail!(
            "kiln-lm-head-kernel: supports only contiguous CUDA BF16 [batch<=32,1,H] x [H,V]"
        );
    }

    let (batch, _, hidden) = x.dims3()?;
    let (_, vocab) = weight_t.dims2()?;
    let device = x.device();
    let scores = Tensor::zeros((batch, vocab), DType::F32, device)?;
    let tokens = Tensor::zeros((batch,), DType::U32, device)?;

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight_t.storage_and_layout();
        let (s_storage, s_layout) = scores.storage_and_layout();
        let (t_storage, t_layout) = tokens.storage_and_layout();

        let x_cuda = match &*x_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-lm-head-kernel: x must be on CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-lm-head-kernel: weight_t must be on CUDA"),
        };
        let s_cuda = match &*s_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-lm-head-kernel: scores must be on CUDA"),
        };
        let t_cuda = match &*t_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-lm-head-kernel: tokens must be on CUDA"),
        };

        let stream = x_cuda.device.cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        let x_slice = x_cuda
            .as_cuda_slice::<bf16>()?
            .slice(x_layout.start_offset()..);
        let w_slice = w_cuda
            .as_cuda_slice::<bf16>()?
            .slice(w_layout.start_offset()..);
        let s_slice = s_cuda
            .as_cuda_slice::<f32>()?
            .slice(s_layout.start_offset()..);
        let t_slice = t_cuda
            .as_cuda_slice::<u32>()?
            .slice(t_layout.start_offset()..);

        unsafe {
            let (x_ptr, _g1) = x_slice.device_ptr(&stream);
            let (w_ptr, _g2) = w_slice.device_ptr(&stream);
            let (s_ptr, _g3) = s_slice.device_ptr(&stream);
            let (t_ptr, _g4) = t_slice.device_ptr(&stream);
            let status = kiln_lm_head_argmax_bf16_batch(
                x_ptr as *const _,
                w_ptr as *const _,
                s_ptr as *mut f32,
                t_ptr as *mut u32,
                batch as i32,
                hidden as i32,
                vocab as i32,
                raw_stream,
            );
            if status != 0 {
                candle_core::bail!("kiln_lm_head_argmax_bf16_batch failed with status {status}");
            }
        }
    }

    tokens.to_vec1::<u32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_cuda_device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    fn deterministic_data(len: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let unit = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
                unit * scale
            })
            .collect()
    }

    fn assert_parity(batch: usize) -> Result<()> {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping CUDA LM-head argmax parity: no CUDA device");
            return Ok(());
        };
        let hidden = 129usize;
        let vocab = 263usize;
        let x_data = deterministic_data(batch * hidden, 0xA11C_E55, 0.75);
        let w_data = deterministic_data(hidden * vocab, 0xBADC_0DE, 0.5);
        let x = Tensor::from_slice(&x_data, (batch, 1usize, hidden), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let weight_t = Tensor::from_slice(&w_data, (hidden, vocab), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;

        assert!(supports_bf16_batch(&x, &weight_t));
        let logits = x.broadcast_matmul(&weight_t)?;
        let reference = crate_argmax_rows(&logits)?;
        let fused = argmax_bf16_batch(&x, &weight_t)?;
        assert_eq!(fused, reference, "batch={batch}");
        Ok(())
    }

    fn crate_argmax_rows(logits: &Tensor) -> Result<Vec<u32>> {
        let (batch, seq_len, _vocab) = logits.dims3()?;
        if seq_len != 1 {
            candle_core::bail!("test helper expects seq_len=1");
        }
        let rows = logits.squeeze(1)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let mut out = Vec::with_capacity(batch);
        for row in rows {
            let mut best_idx = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (idx, score) in row.iter().copied().enumerate() {
                if score > best_score {
                    best_score = score;
                    best_idx = idx;
                }
            }
            out.push(best_idx as u32);
        }
        Ok(out)
    }

    #[test]
    fn test_cuda_lm_head_argmax_batch_1_matches_materialized_logits() -> Result<()> {
        assert_parity(1)
    }

    #[test]
    fn test_cuda_lm_head_argmax_batch_2_matches_materialized_logits() -> Result<()> {
        assert_parity(2)
    }

    #[test]
    fn test_cuda_lm_head_argmax_batch_4_matches_materialized_logits() -> Result<()> {
        assert_parity(4)
    }

    #[test]
    fn test_cuda_lm_head_argmax_batch_8_matches_materialized_logits() -> Result<()> {
        assert_parity(8)
    }
}
