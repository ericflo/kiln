//! Vendored flash-attention-2 CUDA kernels with forward AND backward pass.
//!
//! This crate provides `flash_attn_fwd` and `flash_attn_bwd` functions that operate
//! on candle `Tensor`s, backed by vendored flash-attention CUDA kernels compiled via
//! a thin C-ABI wrapper (no PyTorch dependency).
//!
//! Only bf16, head_dim=128, causal=true instantiations are compiled to minimize build time.

use candle_core::{
    backend::BackendStorage,
    cuda_backend::cudarc::driver::DevicePtr,
    DType, Result, Tensor,
};
use half::bf16;

// FFI declarations matching flash_api_c.h
unsafe extern "C" {
    fn kiln_flash_attn_fwd(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        softmax_lse_out: *mut core::ffi::c_void,
        batch_size: i32,
        seqlen_q: i32,
        seqlen_k: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_dim: i32,
        softmax_scale: f32,
        is_causal: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_flash_attn_bwd(
        dout: *const core::ffi::c_void,
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        out: *const core::ffi::c_void,
        softmax_lse: *const core::ffi::c_void,
        dq: *mut core::ffi::c_void,
        dk: *mut core::ffi::c_void,
        dv: *mut core::ffi::c_void,
        softmax_d_out: *mut core::ffi::c_void,
        dq_accum: *mut core::ffi::c_void,
        batch_size: i32,
        seqlen_q: i32,
        seqlen_k: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_dim: i32,
        softmax_scale: f32,
        is_causal: i32,
        deterministic: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

fn round_up(x: usize, m: usize) -> usize {
    (x + m - 1) / m * m
}

/// Flash-attention v2 forward pass.
///
/// Inputs are `[batch, seqlen, num_heads, head_dim]` bf16 tensors.
/// Returns the attention output tensor with the same shape.
///
/// Also returns softmax_lse `[batch, num_heads, seqlen_q]` as f32 (needed for backward).
pub fn flash_attn_fwd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<(Tensor, Tensor)> {
    let device = q.device();
    let (b, seqlen_q, num_heads, head_dim) = q.dims4()?;
    let (_b, seqlen_k, num_heads_k, _hd) = k.dims4()?;

    if q.dtype() != DType::BF16 {
        candle_core::bail!("kiln-flash-attn only supports bf16, got {:?}", q.dtype());
    }
    if head_dim != 128 && head_dim != 256 {
        candle_core::bail!("kiln-flash-attn only supports head_dim=128,256, got {head_dim}");
    }

    // Ensure contiguous
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;

    // Allocate output
    let out = Tensor::zeros((b, seqlen_q, num_heads, head_dim), DType::BF16, device)?;
    let softmax_lse = Tensor::zeros((b, num_heads, seqlen_q), DType::F32, device)?;

    // Scope the storage borrows so they're dropped before we return the tensors
    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (v_storage, v_layout) = v.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();
        let (lse_storage, lse_layout) = softmax_lse.storage_and_layout();

        let q_cuda = match &*q_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("q must be a CUDA tensor"),
        };
        let k_cuda = match &*k_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("k must be a CUDA tensor"),
        };
        let v_cuda = match &*v_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("v must be a CUDA tensor"),
        };
        let out_cuda = match &*out_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("out must be a CUDA tensor"),
        };
        let lse_cuda = match &*lse_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("softmax_lse must be a CUDA tensor"),
        };

        let stream = q_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let q_slice = q_cuda.as_cuda_slice::<bf16>()?;
        let k_slice = k_cuda.as_cuda_slice::<bf16>()?;
        let v_slice = v_cuda.as_cuda_slice::<bf16>()?;
        let out_slice = out_cuda.as_cuda_slice::<bf16>()?;
        let lse_slice = lse_cuda.as_cuda_slice::<f32>()?;

        let q_slice = q_slice.slice(q_layout.start_offset()..);
        let k_slice = k_slice.slice(k_layout.start_offset()..);
        let v_slice = v_slice.slice(v_layout.start_offset()..);
        let out_slice = out_slice.slice(out_layout.start_offset()..);
        let lse_slice = lse_slice.slice(lse_layout.start_offset()..);

        unsafe {
            let (q_ptr, _g1) = q_slice.device_ptr(&stream);
            let (k_ptr, _g2) = k_slice.device_ptr(&stream);
            let (v_ptr, _g3) = v_slice.device_ptr(&stream);
            let (out_ptr, _g4) = out_slice.device_ptr(&stream);
            let (lse_ptr, _g5) = lse_slice.device_ptr(&stream);

            let status = kiln_flash_attn_fwd(
                q_ptr as *const _,
                k_ptr as *const _,
                v_ptr as *const _,
                out_ptr as *mut _,
                lse_ptr as *mut _,
                b as i32,
                seqlen_q as i32,
                seqlen_k as i32,
                num_heads as i32,
                num_heads_k as i32,
                head_dim as i32,
                softmax_scale,
                if causal { 1 } else { 0 },
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!("kiln_flash_attn_fwd failed with status {status}");
            }
        }
    }

    Ok((out, softmax_lse))
}

/// Flash-attention v2 backward pass.
///
/// Given the gradient of the output (`dout`), the original inputs (`q`, `k`, `v`),
/// the forward output (`out`), and the softmax log-sum-exp (`softmax_lse`),
/// computes gradients `dq`, `dk`, `dv`.
///
/// All bf16 tensors are `[batch, seqlen, num_heads, head_dim]`.
/// `softmax_lse` is `[batch, num_heads, seqlen_q]` f32.
pub fn flash_attn_bwd(
    dout: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &Tensor,
    softmax_lse: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<(Tensor, Tensor, Tensor)> {
    let device = q.device();
    let (b, seqlen_q, num_heads, head_dim) = q.dims4()?;
    let (_b, seqlen_k, num_heads_k, _hd) = k.dims4()?;

    if q.dtype() != DType::BF16 {
        candle_core::bail!("kiln-flash-attn only supports bf16, got {:?}", q.dtype());
    }
    if head_dim != 128 && head_dim != 256 {
        candle_core::bail!("kiln-flash-attn only supports head_dim=128,256, got {head_dim}");
    }

    // Ensure contiguous
    let dout = dout.contiguous()?;
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let out = out.contiguous()?;

    let seqlen_q_rounded = round_up(seqlen_q, 128);
    let head_dim_rounded = round_up(head_dim, 32);

    // Allocate gradient outputs
    let dq = Tensor::zeros((b, seqlen_q, num_heads, head_dim), DType::BF16, device)?;
    // For GQA backward: expand dk/dv to num_heads, then sum later
    let dk = Tensor::zeros((b, seqlen_k, num_heads, head_dim), DType::BF16, device)?;
    let dv = Tensor::zeros((b, seqlen_k, num_heads, head_dim), DType::BF16, device)?;

    // Scratch buffers
    let softmax_d = Tensor::zeros(
        (b, num_heads, seqlen_q_rounded),
        DType::F32,
        device,
    )?;
    let dq_accum = Tensor::zeros(
        (b, seqlen_q_rounded, num_heads, head_dim_rounded),
        DType::F32,
        device,
    )?;

    // Scope the storage borrows so they're dropped before we return/reshape the tensors
    {
        let (dout_s, dout_l) = dout.storage_and_layout();
        let (q_s, q_l) = q.storage_and_layout();
        let (k_s, k_l) = k.storage_and_layout();
        let (v_s, v_l) = v.storage_and_layout();
        let (out_s, out_l) = out.storage_and_layout();
        let (lse_s, lse_l) = softmax_lse.storage_and_layout();
        let (dq_s, dq_l) = dq.storage_and_layout();
        let (dk_s, dk_l) = dk.storage_and_layout();
        let (dv_s, dv_l) = dv.storage_and_layout();
        let (sd_s, sd_l) = softmax_d.storage_and_layout();
        let (da_s, da_l) = dq_accum.storage_and_layout();

        macro_rules! cuda {
            ($s:expr) => {
                match &*$s {
                    candle_core::Storage::Cuda(c) => c,
                    _ => candle_core::bail!("tensor must be on CUDA"),
                }
            };
        }

        let dout_cuda = cuda!(dout_s);
        let q_cuda = cuda!(q_s);
        let k_cuda = cuda!(k_s);
        let v_cuda = cuda!(v_s);
        let out_cuda = cuda!(out_s);
        let lse_cuda = cuda!(lse_s);
        let dq_cuda = cuda!(dq_s);
        let dk_cuda = cuda!(dk_s);
        let dv_cuda = cuda!(dv_s);
        let sd_cuda = cuda!(sd_s);
        let da_cuda = cuda!(da_s);

        let stream = q_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let dout_sl = dout_cuda.as_cuda_slice::<bf16>()?.slice(dout_l.start_offset()..);
        let q_sl = q_cuda.as_cuda_slice::<bf16>()?.slice(q_l.start_offset()..);
        let k_sl = k_cuda.as_cuda_slice::<bf16>()?.slice(k_l.start_offset()..);
        let v_sl = v_cuda.as_cuda_slice::<bf16>()?.slice(v_l.start_offset()..);
        let out_sl = out_cuda.as_cuda_slice::<bf16>()?.slice(out_l.start_offset()..);
        let lse_sl = lse_cuda.as_cuda_slice::<f32>()?.slice(lse_l.start_offset()..);
        let dq_sl = dq_cuda.as_cuda_slice::<bf16>()?.slice(dq_l.start_offset()..);
        let dk_sl = dk_cuda.as_cuda_slice::<bf16>()?.slice(dk_l.start_offset()..);
        let dv_sl = dv_cuda.as_cuda_slice::<bf16>()?.slice(dv_l.start_offset()..);
        let sd_sl = sd_cuda.as_cuda_slice::<f32>()?.slice(sd_l.start_offset()..);
        let da_sl = da_cuda.as_cuda_slice::<f32>()?.slice(da_l.start_offset()..);

        unsafe {
            let (dout_ptr, _g1) = dout_sl.device_ptr(&stream);
            let (q_ptr, _g2) = q_sl.device_ptr(&stream);
            let (k_ptr, _g3) = k_sl.device_ptr(&stream);
            let (v_ptr, _g4) = v_sl.device_ptr(&stream);
            let (out_ptr, _g5) = out_sl.device_ptr(&stream);
            let (lse_ptr, _g6) = lse_sl.device_ptr(&stream);
            let (dq_ptr, _g7) = dq_sl.device_ptr(&stream);
            let (dk_ptr, _g8) = dk_sl.device_ptr(&stream);
            let (dv_ptr, _g9) = dv_sl.device_ptr(&stream);
            let (sd_ptr, _g10) = sd_sl.device_ptr(&stream);
            let (da_ptr, _g11) = da_sl.device_ptr(&stream);

            let status = kiln_flash_attn_bwd(
                dout_ptr as *const _,
                q_ptr as *const _,
                k_ptr as *const _,
                v_ptr as *const _,
                out_ptr as *const _,
                lse_ptr as *const _,
                dq_ptr as *mut _,
                dk_ptr as *mut _,
                dv_ptr as *mut _,
                sd_ptr as *mut _,
                da_ptr as *mut _,
                b as i32,
                seqlen_q as i32,
                seqlen_k as i32,
                num_heads as i32,
                num_heads_k as i32,
                head_dim as i32,
                softmax_scale,
                if causal { 1 } else { 0 },
                0, // non-deterministic
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!("kiln_flash_attn_bwd failed with status {status}");
            }
        }
    }

    // If GQA: dk/dv have num_heads heads, need to sum down to num_heads_k
    if num_heads_k != num_heads {
        let groups = num_heads / num_heads_k;
        // Reshape [b, s, num_heads, d] -> [b, s, num_heads_k, groups, d] and sum over groups dim
        let dk = dk
            .reshape((b, seqlen_k, num_heads_k, groups, head_dim))?
            .sum(3)?;
        let dv = dv
            .reshape((b, seqlen_k, num_heads_k, groups, head_dim))?
            .sum(3)?;
        Ok((dq, dk, dv))
    } else {
        Ok((dq, dk, dv))
    }
}

/// Convenience wrapper matching candle-flash-attn's `flash_attn` API.
/// Returns only the attention output (discards softmax_lse).
pub fn flash_attn(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32, causal: bool) -> Result<Tensor> {
    let (out, _lse) = flash_attn_fwd(q, k, v, softmax_scale, causal)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType, Tensor};

    #[test]
    fn test_flash_attn_forward_basic() {
        let device = Device::new_cuda(0).expect("CUDA device required");
        let b = 1;
        let seqlen = 64;
        let num_heads = 4;
        let head_dim = 128;
        let shape = (b, seqlen, num_heads, head_dim);

        let q = Tensor::randn(0f32, 1.0, shape, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k = Tensor::randn(0f32, 1.0, shape, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let v = Tensor::randn(0f32, 1.0, shape, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        let (out, lse) = flash_attn_fwd(&q, &k, &v, softmax_scale, true).unwrap();

        assert_eq!(out.dims(), &[b, seqlen, num_heads, head_dim]);
        assert_eq!(out.dtype(), DType::BF16);
        assert_eq!(lse.dims(), &[b, num_heads, seqlen]);
        assert_eq!(lse.dtype(), DType::F32);

        // Check output is finite and non-zero
        let out_f32 = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let out_data: Vec<f32> = out_f32.to_vec1().unwrap();
        assert!(out_data.iter().all(|x| x.is_finite()), "output contains non-finite values");
        let abs_sum: f32 = out_data.iter().map(|x| x.abs()).sum();
        assert!(abs_sum > 0.0, "output is all zeros");
        let mean_abs = abs_sum / out_data.len() as f32;
        let max_abs = out_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        eprintln!("fwd output: mean_abs={mean_abs:.6}, max_abs={max_abs:.6}, numel={}", out_data.len());

        // Check LSE is finite
        let lse_f32 = lse.flatten_all().unwrap();
        let lse_data: Vec<f32> = lse_f32.to_vec1().unwrap();
        assert!(lse_data.iter().all(|x| x.is_finite()), "softmax_lse contains non-finite values");
        eprintln!("fwd softmax_lse: min={:.4}, max={:.4}",
            lse_data.iter().cloned().fold(f32::INFINITY, f32::min),
            lse_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    }

    #[test]
    fn test_flash_attn_backward_gradients() {
        let device = Device::new_cuda(0).expect("CUDA device required");
        let b = 1;
        let seqlen = 64;
        let num_heads = 4;
        let num_heads_k = 4; // MHA (not GQA) for simplicity
        let head_dim = 128;

        let q = Tensor::randn(0f32, 1.0, (b, seqlen, num_heads, head_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k = Tensor::randn(0f32, 1.0, (b, seqlen, num_heads_k, head_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let v = Tensor::randn(0f32, 1.0, (b, seqlen, num_heads_k, head_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let softmax_scale = 1.0 / (head_dim as f32).sqrt();

        // Forward pass
        let (out, softmax_lse) = flash_attn_fwd(&q, &k, &v, softmax_scale, true).unwrap();

        // Simulate gradient: dout = ones (uniform gradient signal)
        let dout = Tensor::ones((b, seqlen, num_heads, head_dim), DType::BF16, &device).unwrap();

        // Backward pass
        let (dq, dk, dv) = flash_attn_bwd(&dout, &q, &k, &v, &out, &softmax_lse, softmax_scale, true).unwrap();

        // Check shapes
        assert_eq!(dq.dims(), &[b, seqlen, num_heads, head_dim]);
        assert_eq!(dk.dims(), &[b, seqlen, num_heads_k, head_dim]);
        assert_eq!(dv.dims(), &[b, seqlen, num_heads_k, head_dim]);

        // Check all gradients are finite, non-zero, and reasonable magnitude
        for (name, grad) in [("dq", &dq), ("dk", &dk), ("dv", &dv)] {
            let data: Vec<f32> = grad
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert!(
                data.iter().all(|x| x.is_finite()),
                "{name} contains non-finite values"
            );
            let abs_sum: f32 = data.iter().map(|x| x.abs()).sum();
            assert!(abs_sum > 0.0, "{name} is all zeros");
            let mean_abs = abs_sum / data.len() as f32;
            let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!("{name}: mean_abs={mean_abs:.6}, max_abs={max_abs:.6}, numel={}", data.len());
            assert!(max_abs < 100.0, "{name} has unreasonably large gradient: max_abs={max_abs}");
        }
    }

    #[test]
    fn test_flash_attn_backward_gqa() {
        let device = Device::new_cuda(0).expect("CUDA device required");
        let b = 1;
        let seqlen = 32;
        let num_heads = 8;
        let num_heads_k = 2; // GQA: 4 groups
        let head_dim = 128;

        let q = Tensor::randn(0f32, 1.0, (b, seqlen, num_heads, head_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k = Tensor::randn(0f32, 1.0, (b, seqlen, num_heads_k, head_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let v = Tensor::randn(0f32, 1.0, (b, seqlen, num_heads_k, head_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let softmax_scale = 1.0 / (head_dim as f32).sqrt();

        // Forward: expand K/V to num_heads for the kernel
        let groups = num_heads / num_heads_k;
        let k_exp = k
            .unsqueeze(3)
            .unwrap()
            .expand((b, seqlen, num_heads_k, groups, head_dim))
            .unwrap()
            .reshape((b, seqlen, num_heads, head_dim))
            .unwrap()
            .contiguous()
            .unwrap();
        let v_exp = v
            .unsqueeze(3)
            .unwrap()
            .expand((b, seqlen, num_heads_k, groups, head_dim))
            .unwrap()
            .reshape((b, seqlen, num_heads, head_dim))
            .unwrap()
            .contiguous()
            .unwrap();

        let (out, softmax_lse) =
            flash_attn_fwd(&q, &k_exp, &v_exp, softmax_scale, true).unwrap();

        let dout = Tensor::ones((b, seqlen, num_heads, head_dim), DType::BF16, &device).unwrap();

        // Backward with expanded K/V (same as how model code calls it)
        // Since k_exp/v_exp have num_heads heads, bwd sees num_heads_k == num_heads
        // and outputs dk/dv with num_heads (caller is responsible for summing to GQA groups)
        let (dq, dk, dv) =
            flash_attn_bwd(&dout, &q, &k_exp, &v_exp, &out, &softmax_lse, softmax_scale, true)
                .unwrap();

        assert_eq!(dq.dims(), &[b, seqlen, num_heads, head_dim]);
        assert_eq!(dk.dims(), &[b, seqlen, num_heads, head_dim]);
        assert_eq!(dv.dims(), &[b, seqlen, num_heads, head_dim]);

        for (name, grad) in [("dq", &dq), ("dk", &dk), ("dv", &dv)] {
            let data: Vec<f32> = grad
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert!(
                data.iter().all(|x| x.is_finite()),
                "{name} contains non-finite values"
            );
            let abs_sum: f32 = data.iter().map(|x| x.abs()).sum();
            assert!(abs_sum > 0.0, "{name} is all zeros");
        }
    }
}
