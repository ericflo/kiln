//! Vendored mamba-ssm `causal_conv1d_update` CUDA kernel, narrowed to kiln's
//! Qwen3.5-4B GDN decode envelope.
//!
//! # Why
//!
//! `kiln-model::forward::causal_conv1d_decode` expresses the single-step
//! depthwise conv1d + state update as a chain of candle ops:
//!
//! ```text
//! x_f32     = to_dtype(F32)(x)
//! w_f32     = to_dtype(F32)(weight).reshape((C, K))
//! state_f32 = to_dtype(F32)(conv_state)
//! window    = cat(state_f32, x_f32, dim=2)          // [B, C, K]
//! out       = sum(window * w.unsqueeze(0), dim=2)   // [B, C]
//! conv_state = window.narrow(2, 1, K-1).contiguous() // [B, C, K-1]
//! out       = silu(out.unsqueeze(2))                 // applied by caller
//! ```
//!
//! That is ~6 CUDA launches per GDN layer × 24 layers ≈ 144 launches per
//! decode step just for the conv. Per PROFILING.md (post-PR #158) the
//! `kiln/gdn/conv` NVTX region is **12.2 %** of decode wall-clock.
//!
//! This crate collapses the whole chain into a single launch:
//! `kiln_causal_conv1d_update_bf16_f32` — one thread per (batch, channel),
//! K=4 registers for the window, F32 accumulator, F32 state, F32 silu
//! epilogue.
//!
//! # Provenance
//!
//! Algorithm vendored from
//! [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)
//! (mamba-ssm) `csrc/causal_conv1d_update.cu`, kIsCircularBuffer=false,
//! kNBytes=2, kWidth=4 specialisation. License: Apache 2.0 upstream; this
//! crate retains the same licence for the vendored portion.
//!
//! # Scope
//!
//! - Decode single-step only (`seq_len == 1`).
//! - bf16 activations, bf16 weights, F32 state, F32 output.
//! - `kernel_size == 4` (Qwen3.5 GDN). Other widths return `Ok(None)`.
//! - No bias (kiln doesn't load conv1d bias — see forward.rs line ~1478).
//! - No per-batch conv_state_indices / circular buffer.
//! - SiLU fused inline (matches the `cuda_silu` call immediately after the
//!   conv in `gated_deltanet_forward`).
//!
//! # API
//!
//! - [`supports`] — cheap envelope check for the backend trait's
//!   `supports_causal_conv1d_update` hook.
//! - [`causal_conv1d_update`] — candle-compatible entry point. Mutates
//!   `conv_state` in place (matches the candle fallback's semantics).

use candle_core::{
    DType, Device, Result, Tensor, backend::BackendStorage, cuda_backend::cudarc::driver::DevicePtr,
};
use half::bf16;

unsafe extern "C" {
    fn kiln_causal_conv1d_update_bf16_f32(
        x: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        conv_state: *mut core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch: i32,
        channels: i32,
        kernel_width: i32,
        silu: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_causal_conv1d_prefill_bf16_f32(
        x: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        conv_state: *mut core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch: i32,
        channels: i32,
        seq_len: i32,
        kernel_width: i32,
        silu: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

fn conv1d_status_description(status: i32) -> &'static str {
    match status {
        1 => "unsupported kernel width (only kernel_size=4 is compiled)",
        2 => "invalid launch dimensions",
        3 => "CUDA launch failed; check the kernel launch bounds and requested block size",
        _ => "unknown CUDA conv1d kernel failure",
    }
}

/// Whether the fused kernel can handle this call.
///
/// Returns `true` only for the exact bf16/f32/K=4 envelope the vendored
/// kernel was specialised for. All other shapes/dtypes should take the
/// candle fallback path.
pub fn supports(x: &Tensor, weight: &Tensor, conv_state: &Tensor, kernel_size: usize) -> bool {
    supports_update(x, weight, conv_state, kernel_size)
}

/// Whether the fused single-token update kernel can handle this call.
pub fn supports_update(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
) -> bool {
    if kernel_size != 4 {
        return false;
    }
    if !matches!(x.device(), Device::Cuda(_)) {
        return false;
    }
    if x.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return false;
    }
    if conv_state.dtype() != DType::F32 {
        return false;
    }
    // x: [B, C, 1]
    let Ok((batch, channels, t)) = x.dims3() else {
        return false;
    };
    if t != 1 {
        return false;
    }
    // conv_state: [B, C, K-1]
    let Ok((bs, cs, ks)) = conv_state.dims3() else {
        return false;
    };
    if (bs, cs, ks) != (batch, channels, kernel_size - 1) {
        return false;
    }
    weight_supports(weight, channels, kernel_size)
}

/// Whether the fused multi-token prefill kernel can handle this call.
pub fn supports_prefill(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
) -> bool {
    if kernel_size != 4 {
        return false;
    }
    if !matches!(x.device(), Device::Cuda(_)) {
        return false;
    }
    if x.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return false;
    }
    if conv_state.dtype() != DType::F32 {
        return false;
    }
    // x: [B, C, T], T > 1
    let Ok((batch, channels, t)) = x.dims3() else {
        return false;
    };
    if t <= 1 {
        return false;
    }
    // conv_state: [B, C, K-1]
    let Ok((bs, cs, ks)) = conv_state.dims3() else {
        return false;
    };
    if (bs, cs, ks) != (batch, channels, kernel_size - 1) {
        return false;
    }
    weight_supports(weight, channels, kernel_size)
}

fn weight_supports(weight: &Tensor, channels: usize, kernel_size: usize) -> bool {
    match weight.rank() {
        3 => weight
            .dims3()
            .is_ok_and(|(c, one, k)| c == channels && one == 1 && k == kernel_size),
        2 => weight
            .dims2()
            .is_ok_and(|(c, k)| c == channels && k == kernel_size),
        _ => false,
    }
}

/// Run the fused causal_conv1d_update kernel.
///
/// Inputs:
///   - `x`: bf16 `[B, C, 1]` — current token per channel (depthwise).
///   - `weight`: bf16 `[C, 1, K]` (or `[C, K]` after squeeze) — conv weights.
///   - `conv_state`: F32 `[B, C, K-1]` — rolling buffer, mutated in place.
///   - `kernel_size`: must be 4.
///
/// Output: F32 `[B, C, 1]` with SiLU fused inline. On return, `conv_state`
/// has been updated to drop the oldest sample and append `x`.
pub fn causal_conv1d_update(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    if kernel_size != 4 {
        candle_core::bail!(
            "kiln-conv1d-kernel: only kernel_size=4 is supported (got {kernel_size})"
        );
    }

    let device = x.device();

    let (batch, channels, t) = x.dims3()?;
    if t != 1 {
        candle_core::bail!("kiln-conv1d-kernel: decode path requires seq_len=1 (got {t})");
    }

    // Accept either [C, 1, K] or [C, K] for weight (both are how kiln expresses
    // depthwise conv weights).
    let weight_flat = match weight.rank() {
        3 => {
            let (c, one, k) = weight.dims3()?;
            if c != channels || one != 1 || k != kernel_size {
                candle_core::bail!(
                    "kiln-conv1d-kernel: weight shape [{c}, {one}, {k}] does not match channels={channels} kernel_size={kernel_size}"
                );
            }
            weight.reshape((channels, kernel_size))?
        }
        2 => {
            let (c, k) = weight.dims2()?;
            if c != channels || k != kernel_size {
                candle_core::bail!(
                    "kiln-conv1d-kernel: weight shape [{c}, {k}] does not match channels={channels} kernel_size={kernel_size}"
                );
            }
            weight.clone()
        }
        r => {
            candle_core::bail!("kiln-conv1d-kernel: weight must be rank 2 or 3 (got rank {r})")
        }
    };

    let (bs, cs, ks) = conv_state.dims3()?;
    if (bs, cs, ks) != (batch, channels, kernel_size - 1) {
        candle_core::bail!(
            "kiln-conv1d-kernel: conv_state shape [{bs}, {cs}, {ks}] mismatch with [{batch}, {channels}, {}] ",
            kernel_size - 1
        );
    }

    if x.dtype() != DType::BF16 {
        candle_core::bail!("kiln-conv1d-kernel: x must be bf16 (got {:?})", x.dtype());
    }
    if weight_flat.dtype() != DType::BF16 {
        candle_core::bail!(
            "kiln-conv1d-kernel: weight must be bf16 (got {:?})",
            weight_flat.dtype()
        );
    }
    if conv_state.dtype() != DType::F32 {
        candle_core::bail!(
            "kiln-conv1d-kernel: conv_state must be f32 (got {:?})",
            conv_state.dtype()
        );
    }

    let x = x.contiguous()?;
    let weight_flat = weight_flat.contiguous()?;
    if !conv_state.is_contiguous() {
        *conv_state = conv_state.contiguous()?;
    }

    let out = Tensor::zeros((batch, channels, 1), DType::F32, device)?;

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight_flat.storage_and_layout();
        let (s_storage, s_layout) = conv_state.storage_and_layout();
        let (o_storage, o_layout) = out.storage_and_layout();

        let x_cuda = match &*x_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-conv1d-kernel: x must be on CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-conv1d-kernel: weight must be on CUDA"),
        };
        let s_cuda = match &*s_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-conv1d-kernel: conv_state must be on CUDA"),
        };
        let o_cuda = match &*o_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-conv1d-kernel: out must be on CUDA"),
        };

        let stream = x_cuda.device().cuda_stream();
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
        let o_slice = o_cuda
            .as_cuda_slice::<f32>()?
            .slice(o_layout.start_offset()..);

        unsafe {
            let (x_ptr, _g1) = x_slice.device_ptr(&stream);
            let (w_ptr, _g2) = w_slice.device_ptr(&stream);
            let (s_ptr, _g3) = s_slice.device_ptr(&stream);
            let (o_ptr, _g4) = o_slice.device_ptr(&stream);

            let status = kiln_causal_conv1d_update_bf16_f32(
                x_ptr as *const _,
                w_ptr as *const _,
                s_ptr as *mut _,
                o_ptr as *mut _,
                batch as i32,
                channels as i32,
                kernel_size as i32,
                1, // silu=true — matches the cuda_silu call immediately after in forward.rs
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!(
                    "kiln_causal_conv1d_update_bf16_f32 failed with status {status}: {} (batch={batch}, channels={channels}, kernel_size={kernel_size})",
                    conv1d_status_description(status)
                );
            }
        }
    }

    Ok(out)
}

/// Run the fused causal_conv1d prefill kernel.
///
/// Inputs:
///   - `x`: bf16 `[B, C, T]`, `T > 1`.
///   - `weight`: bf16 `[C, 1, K]` (or `[C, K]`) conv weights.
///   - `conv_state`: F32 `[B, C, K-1]` — entry state, mutated after output.
///   - `kernel_size`: must be 4.
///
/// Output: F32 `[B, C, T]` with SiLU fused inline. On return,
/// `conv_state` contains the last K-1 input positions exactly like the
/// portable prefill fallback.
pub fn causal_conv1d_prefill(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    if kernel_size != 4 {
        candle_core::bail!(
            "kiln-conv1d-kernel: only kernel_size=4 is supported (got {kernel_size})"
        );
    }

    let device = x.device();
    let (batch, channels, seq_len) = x.dims3()?;
    if seq_len <= 1 {
        candle_core::bail!("kiln-conv1d-kernel: prefill path requires seq_len > 1 (got {seq_len})");
    }

    let weight_flat = match weight.rank() {
        3 => {
            let (c, one, k) = weight.dims3()?;
            if c != channels || one != 1 || k != kernel_size {
                candle_core::bail!(
                    "kiln-conv1d-kernel: weight shape [{c}, {one}, {k}] does not match channels={channels} kernel_size={kernel_size}"
                );
            }
            weight.reshape((channels, kernel_size))?
        }
        2 => {
            let (c, k) = weight.dims2()?;
            if c != channels || k != kernel_size {
                candle_core::bail!(
                    "kiln-conv1d-kernel: weight shape [{c}, {k}] does not match channels={channels} kernel_size={kernel_size}"
                );
            }
            weight.clone()
        }
        r => {
            candle_core::bail!("kiln-conv1d-kernel: weight must be rank 2 or 3 (got rank {r})")
        }
    };

    let (bs, cs, ks) = conv_state.dims3()?;
    if (bs, cs, ks) != (batch, channels, kernel_size - 1) {
        candle_core::bail!(
            "kiln-conv1d-kernel: conv_state shape [{bs}, {cs}, {ks}] mismatch with [{batch}, {channels}, {}] ",
            kernel_size - 1
        );
    }

    if x.dtype() != DType::BF16 {
        candle_core::bail!("kiln-conv1d-kernel: x must be bf16 (got {:?})", x.dtype());
    }
    if weight_flat.dtype() != DType::BF16 {
        candle_core::bail!(
            "kiln-conv1d-kernel: weight must be bf16 (got {:?})",
            weight_flat.dtype()
        );
    }
    if conv_state.dtype() != DType::F32 {
        candle_core::bail!(
            "kiln-conv1d-kernel: conv_state must be f32 (got {:?})",
            conv_state.dtype()
        );
    }

    let x = x.contiguous()?;
    let weight_flat = weight_flat.contiguous()?;
    if !conv_state.is_contiguous() {
        *conv_state = conv_state.contiguous()?;
    }

    let out = Tensor::zeros((batch, channels, seq_len), DType::F32, device)?;

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight_flat.storage_and_layout();
        let (s_storage, s_layout) = conv_state.storage_and_layout();
        let (o_storage, o_layout) = out.storage_and_layout();

        let x_cuda = match &*x_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-conv1d-kernel: x must be on CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-conv1d-kernel: weight must be on CUDA"),
        };
        let s_cuda = match &*s_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-conv1d-kernel: conv_state must be on CUDA"),
        };
        let o_cuda = match &*o_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-conv1d-kernel: out must be on CUDA"),
        };

        let stream = x_cuda.device().cuda_stream();
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
        let o_slice = o_cuda
            .as_cuda_slice::<f32>()?
            .slice(o_layout.start_offset()..);

        unsafe {
            let (x_ptr, _g1) = x_slice.device_ptr(&stream);
            let (w_ptr, _g2) = w_slice.device_ptr(&stream);
            let (s_ptr, _g3) = s_slice.device_ptr(&stream);
            let (o_ptr, _g4) = o_slice.device_ptr(&stream);

            let status = kiln_causal_conv1d_prefill_bf16_f32(
                x_ptr as *const _,
                w_ptr as *const _,
                s_ptr as *mut _,
                o_ptr as *mut _,
                batch as i32,
                channels as i32,
                seq_len as i32,
                kernel_size as i32,
                1,
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!(
                    "kiln_causal_conv1d_prefill_bf16_f32 failed with status {status}: {} (batch={batch}, channels={channels}, seq_len={seq_len}, kernel_size={kernel_size})",
                    conv1d_status_description(status)
                );
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    // Reference implementation — mirrors `kiln-model::forward::causal_conv1d_decode`
    // followed by the `cuda_silu(out.to_dtype(F32))` the caller applies.
    fn reference_decode_with_silu(
        x: &Tensor,
        weight: &Tensor,
        conv_state: &mut Tensor,
        kernel_size: usize,
    ) -> Result<Tensor> {
        let (_batch, channels, _one) = x.dims3()?;
        let x_f32 = x.to_dtype(DType::F32)?;
        let w_f32 = weight
            .to_dtype(DType::F32)?
            .reshape((channels, kernel_size))?;
        let window = Tensor::cat(&[&conv_state.clone(), &x_f32], 2)?;
        let w_expanded = w_f32.unsqueeze(0)?;
        let output = window.broadcast_mul(&w_expanded)?.sum(2)?.unsqueeze(2)?;
        *conv_state = window.narrow(2, 1, kernel_size - 1)?.contiguous()?;
        // SiLU: x / (1 + exp(-x))
        let ones = output.ones_like()?;
        let silu = (&output / (ones.clone() + output.neg()?.exp()?)?)?;
        Ok(silu)
    }

    fn reference_prefill_with_silu(
        x: &Tensor,
        weight: &Tensor,
        conv_state: &mut Tensor,
        kernel_size: usize,
    ) -> Result<Tensor> {
        let (_batch, channels, seq_len) = x.dims3()?;
        let x_f32 = x.to_dtype(DType::F32)?;
        let w_f32 = weight
            .to_dtype(DType::F32)?
            .reshape((channels, kernel_size))?;
        let x_padded = Tensor::cat(&[&conv_state.clone(), &x_f32], 2)?;
        let mut outs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let window = x_padded.narrow(2, t, kernel_size)?;
            let out_t = window
                .broadcast_mul(&w_f32.unsqueeze(0)?)?
                .sum(2)?
                .unsqueeze(2)?;
            outs.push(out_t);
        }
        let outs_ref: Vec<&Tensor> = outs.iter().collect();
        let output = Tensor::cat(&outs_ref, 2)?;
        *conv_state = if seq_len >= kernel_size - 1 {
            x_f32
                .narrow(2, seq_len - (kernel_size - 1), kernel_size - 1)?
                .contiguous()?
        } else {
            let keep = kernel_size - 1 - seq_len;
            let old_part = conv_state.narrow(2, seq_len, keep)?;
            Tensor::cat(&[&old_part, &x_f32], 2)?.contiguous()?
        };
        let ones = output.ones_like()?;
        let silu = (&output / (ones.clone() + output.neg()?.exp()?)?)?;
        Ok(silu)
    }

    fn try_cuda_device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    fn seeded(n: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            v.push(f * scale);
        }
        v
    }

    #[test]
    fn parity_qwen35_decode_shape() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // Qwen3.5-4B decode: batch=1, conv_dim=linear_qkv_dim()=8192, K=4.
        let batch = 1;
        let channels = 8192;
        let kernel_size = 4;

        let raw_x = seeded(batch * channels * 1, 0x1234_5678, 0.5);
        let raw_w = seeded(channels * kernel_size, 0xfeed_face, 0.1);
        let raw_state = seeded(batch * channels * (kernel_size - 1), 0xcafe_babe, 0.3);

        let x_f32 = Tensor::from_vec(raw_x, (batch, channels, 1), &device).unwrap();
        let w_f32 = Tensor::from_vec(raw_w, (channels, 1, kernel_size), &device).unwrap();
        let state_ref_f32 = Tensor::from_vec(
            raw_state.clone(),
            (batch, channels, kernel_size - 1),
            &device,
        )
        .unwrap();

        let x = x_f32.to_dtype(DType::BF16).unwrap();
        let w = w_f32.to_dtype(DType::BF16).unwrap();
        let mut state_ref = state_ref_f32.clone();
        let mut state_fused = state_ref_f32.clone();

        let y_ref = reference_decode_with_silu(&x, &w, &mut state_ref, kernel_size).unwrap();
        let y_fused = causal_conv1d_update(&x, &w, &mut state_fused, kernel_size).unwrap();

        // Output parity.
        let diff = (&y_ref - &y_fused)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            diff < 2e-3,
            "output parity failed: max_abs_diff={diff} exceeds 2e-3 tolerance"
        );

        // State parity — both paths write the same K-1 previous inputs.
        let state_diff = (&state_ref - &state_fused)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            state_diff < 1e-5,
            "conv_state parity failed: max_abs_diff={state_diff} exceeds 1e-5"
        );
    }

    #[test]
    fn parity_qwen35_prefill_shape() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        let batch = 1;
        let channels = 8192;
        let seq_len = 512;
        let kernel_size = 4;

        let raw_x = seeded(batch * channels * seq_len, 0x1234_9876, 0.5);
        let raw_w = seeded(channels * kernel_size, 0xface_feed, 0.1);
        let raw_state = seeded(batch * channels * (kernel_size - 1), 0xbeef_cafe, 0.3);

        let x_f32 = Tensor::from_vec(raw_x, (batch, channels, seq_len), &device).unwrap();
        let w_f32 = Tensor::from_vec(raw_w, (channels, 1, kernel_size), &device).unwrap();
        let state_ref_f32 = Tensor::from_vec(
            raw_state.clone(),
            (batch, channels, kernel_size - 1),
            &device,
        )
        .unwrap();

        let x = x_f32.to_dtype(DType::BF16).unwrap();
        let w = w_f32.to_dtype(DType::BF16).unwrap();
        let mut state_ref = state_ref_f32.clone();
        let mut state_fused = state_ref_f32.clone();

        let y_ref = reference_prefill_with_silu(&x, &w, &mut state_ref, kernel_size).unwrap();
        let y_fused = causal_conv1d_prefill(&x, &w, &mut state_fused, kernel_size).unwrap();

        let diff = (&y_ref - &y_fused)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            diff < 2e-3,
            "prefill output parity failed: max_abs_diff={diff} exceeds 2e-3"
        );

        let state_diff = (&state_ref - &state_fused)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            state_diff < 1e-5,
            "prefill conv_state parity failed: max_abs_diff={state_diff} exceeds 1e-5"
        );
    }

    #[test]
    fn supports_rejects_wrong_width() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        let x = Tensor::zeros((1, 16, 1), DType::BF16, &device).unwrap();
        let w = Tensor::zeros((16, 1, 3), DType::BF16, &device).unwrap();
        let s = Tensor::zeros((1, 16, 2), DType::F32, &device).unwrap();
        assert!(!supports(&x, &w, &s, 3));
    }

    #[test]
    fn supports_accepts_qwen_shape() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        let x = Tensor::zeros((1, 8192, 1), DType::BF16, &device).unwrap();
        let w = Tensor::zeros((8192, 1, 4), DType::BF16, &device).unwrap();
        let s = Tensor::zeros((1, 8192, 3), DType::F32, &device).unwrap();
        assert!(supports(&x, &w, &s, 4));

        let x_prefill = Tensor::zeros((1, 8192, 16), DType::BF16, &device).unwrap();
        assert!(supports_prefill(&x_prefill, &w, &s, 4));
    }
}
