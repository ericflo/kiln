//! `candle_core::CustomOp3` wrapper for Vulkan-dispatched LoRA delta application.
//!
//! Phase 4.1 of the residency plan. Implements `candle_core::CustomOp3` so the
//! on-device LoRA delta dispatch carries an analytic backward that
//! returns gradients for x, A, and B — making it safe for both
//! training-time and inference-time forward passes. (An earlier
//! version returned a candle-leaf candle_core::Tensor with no autograd back-link
//! and had to be gated to inference-only; this candle_core::CustomOp3 wrapper
//! removed that gate.)
//!
//! Forward:
//!   delta = (x @ A.T @ B.T) * scale
//!         = (h @ B.T) * scale       where h = x @ A.T
//!         = d * scale                where d = h @ B.T
//!
//! Backward (given grad_y = dL/d(delta)):
//!   grad_d = grad_y * scale
//!   grad_h = grad_d @ B          (B treated as [out, rank])
//!   grad_x = grad_h @ A          (A treated as [rank, in])
//!   grad_A = grad_h.T @ x        (sum over batch dim → [rank, in])
//!   grad_B = grad_d.T @ h        (sum over batch dim → [out, rank])
//!
//! Forward path: dispatches the existing transposed bf16-packed kernel
//! twice against A's and B's registry buffers. Backward path: reads A
//! and B values directly from those same registry buffers (so the
//! op no longer depends on candle CPU storage of the LoRA Vars —
//! enabling the lazy `sync_to_candle` flow where candle storage is
//! only refreshed before save_peft), then runs the analytic gradient
//! matmuls on candle CPU. The bwd matmuls are tiny (rank≤64) so CPU
//! is fine; the win is removing the every-step `var.set` readback of
//! the optimizer step into candle storage.

use std::sync::Arc;

use anyhow::{Context, Result as AnyResult};
use kiln_vulkan_kernel::{VulkanBuffer, VulkanDevice, kernels};

/// File-private candle⇔bytes helpers — migrated inline from
/// `kiln_vulkan_kernel::kernels::{extract_tensor_bytes,
/// create_tensor_from_data, buffer_to_tensor, upload_tensor_bf16_packed_buffer}`
/// as part of issue #1082 (drop candle from kiln-vulkan-kernel).
///
/// Mirrors the public bridge implementations exactly so this file can
/// perform candle ↔ raw-bytes conversions without routing through the
/// kiln-vulkan-kernel candle bridge surface. (#1082)
#[inline]
fn tensor_to_f32_bytes(tensor: &candle_core::Tensor) -> AnyResult<Vec<u8>> {
    let flat = tensor
        .flatten_all()
        .context("failed to flatten tensor")?;
    let f32_data = flat
        .to_dtype(candle_core::DType::F32)?
        .to_vec1::<f32>()
        .context("failed to extract f32 data")?;
    Ok(bytemuck::cast_slice(&f32_data).to_vec())
}

#[inline]
fn tensor_from_f32_bytes(
    data: &[u8],
    shape: &[usize],
    dtype: candle_core::DType,
) -> AnyResult<candle_core::Tensor> {
    let f32_data: &[f32] = bytemuck::cast_slice(data);
    let tensor = candle_core::Tensor::from_vec(
        f32_data.to_vec(),
        f32_data.len(),
        &candle_core::Device::Cpu,
    )?
    .reshape(shape)?;
    if dtype == candle_core::DType::BF16 {
        Ok(tensor.to_dtype(candle_core::DType::BF16)?)
    } else {
        Ok(tensor)
    }
}

/// Inlined replacement for `kernels::buffer_to_tensor`. Reads back the
/// VulkanBuffer via its host-visible memory and reconstructs the
/// candle tensor — BF16 buffers are stored as packed bf16 (two lanes
/// per u32, `(hi << 16) | lo`); F32 buffers are stored as raw f32
/// bytes. (#1082)
#[inline]
fn buffer_to_tensor_inline(
    vk_device: &VulkanDevice,
    buffer: &VulkanBuffer,
    shape: &[usize],
    dtype: candle_core::DType,
) -> AnyResult<candle_core::Tensor> {
    let bytes = VulkanBuffer::read_back(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        buffer,
    )
    .context("buffer_to_tensor_inline: VulkanBuffer::read_back")?;
    if dtype == candle_core::DType::BF16 {
        anyhow::ensure!(
            bytes.len() % 2 == 0,
            "buffer_to_tensor_inline BF16: buffer byte count {} is not a multiple of 2",
            bytes.len()
        );
        let elem_count: usize = shape.iter().product();
        let stored = bytes.len() / 2;
        anyhow::ensure!(
            stored >= elem_count,
            "buffer_to_tensor_inline BF16: buffer holds {} bf16 elements, expected at least {} \
             for shape {:?}",
            stored,
            elem_count,
            shape,
        );
        let mut f32_data = Vec::with_capacity(elem_count);
        for i in 0..elem_count {
            let lo = bytes[i * 2] as u32;
            let hi = bytes[i * 2 + 1] as u32;
            let bf16_bits = (hi << 8) | lo;
            f32_data.push(f32::from_bits(bf16_bits << 16));
        }
        Ok(candle_core::Tensor::from_vec(f32_data, shape, &candle_core::Device::Cpu)?
            .to_dtype(candle_core::DType::BF16)?)
    } else {
        tensor_from_f32_bytes(&bytes, shape, dtype)
    }
}

/// Inlined replacement for `kernels::upload_tensor_bf16_packed_buffer`.
/// Extracts the tensor's bf16 values then uploads via the candle-free
/// `kernels::upload_bf16_packed_buffer_from_slice`. (#1082)
#[inline]
fn upload_tensor_bf16_packed_buffer_inline(
    vk_device: &VulkanDevice,
    tensor: &candle_core::Tensor,
) -> AnyResult<VulkanBuffer> {
    anyhow::ensure!(
        tensor.dtype() == candle_core::DType::BF16,
        "packed bf16 upload requires BF16 tensor, got {:?}",
        tensor.dtype()
    );
    let bf16_data: Vec<half::bf16> = tensor
        .flatten_all()
        .context("failed to flatten bf16 tensor for upload")?
        .to_vec1::<half::bf16>()
        .context("failed to extract bf16 data for upload")?;
    kernels::upload_bf16_packed_buffer_from_slice(vk_device, &bf16_data)
}

/// Op state for [`VulkanLoraOp`]. Captures the device handle plus the
/// two registry-resident weight buffers (A and B) so `cpu_fwd` can
/// dispatch without re-reading the candle storage.
pub struct VulkanLoraOp {
    pub vk_device: Arc<VulkanDevice>,
    pub a_buffer: Arc<VulkanBuffer>,
    pub b_buffer: Arc<VulkanBuffer>,
    pub rank: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub scale: f32,
    /// Output dtype — typically the input dtype (BF16 in production).
    pub out_dtype: candle_core::DType,
}

impl std::fmt::Debug for VulkanLoraOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanLoraOp")
            .field("rank", &self.rank)
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("scale", &self.scale)
            .field("out_dtype", &self.out_dtype)
            .finish()
    }
}

impl candle_core::CustomOp3 for VulkanLoraOp {
    fn name(&self) -> &'static str {
        "kiln-vulkan-lora-delta"
    }

    fn cpu_fwd(
        &self,
        s_x: &candle_core::CpuStorage,
        l_x: &candle_core::Layout,
        _s_a: &candle_core::CpuStorage,
        _l_a: &candle_core::Layout,
        _s_b: &candle_core::CpuStorage,
        _l_b: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        // We dispatch from registry buffers (kept in sync with A/B
        // candle storage by `update_resident_activation` after each
        // SGD step). The s_a / s_b storages aren't read here; they're
        // accepted to make A and B autograd inputs.
        let x_dims = l_x.shape().dims();
        if x_dims.is_empty() {
            return Err(candle_core::Error::Msg(
                "VulkanLoraOp: x must have at least one dim".into(),
            ));
        }
        let inner = x_dims[x_dims.len() - 1];
        if inner != self.in_features {
            return Err(candle_core::Error::Msg(format!(
                "VulkanLoraOp: x last dim {inner} != in_features {}",
                self.in_features
            )));
        }
        let row_count: usize = x_dims[..x_dims.len() - 1].iter().product();
        if row_count == 0 {
            // Degenerate empty batch — return empty tensor of correct
            // shape rather than dispatching.
            let mut out_dims: Vec<usize> = x_dims[..x_dims.len() - 1].to_vec();
            out_dims.push(self.out_features);
            let zero = candle_core::Tensor::zeros(
                out_dims.as_slice(),
                self.out_dtype,
                &candle_core::Device::Cpu,
            )
            .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp empty zeros: {e:?}")))?;
            let storage = zero
                .storage_and_layout()
                .0
                .try_clone(zero.layout())
                .map_err(|e| {
                    candle_core::Error::Msg(format!("VulkanLoraOp empty storage clone: {e:?}"))
                })?;
            let cpu_storage = match storage {
                candle_core::Storage::Cpu(s) => s,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "VulkanLoraOp: expected CPU storage from empty zeros".into(),
                    ));
                }
            };
            return Ok((cpu_storage, candle_core::Shape::from(out_dims.as_slice())));
        }

        // Wrap the x storage in a candle_core::Tensor briefly so we can extract its
        // raw f32 bytes via `extract_tensor_bytes`. After this the two
        // matmul dispatches go through the candle-free `_bytes` entry
        // points, so the intermediate hidden activation never has to
        // round-trip through candle storage. (#1082)
        let storage = candle_core::Storage::Cpu(s_x.clone());
        let x_tensor = candle_core::Tensor::from_storage(
            storage,
            candle_core::Shape::from(x_dims.to_vec()),
            candle_core::op::BackpropOp::none(),
            false,
        );
        let x_f32 = if x_tensor.dtype() == candle_core::DType::F32 {
            x_tensor
        } else {
            x_tensor
                .to_dtype(candle_core::DType::F32)
                .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp x→f32: {e:?}")))?
        };
        let x_2d = x_f32
            .reshape((row_count, self.in_features))
            .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp reshape x: {e:?}")))?
            .contiguous()
            .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp x contiguous: {e:?}")))?;
        let x_2d_bytes = tensor_to_f32_bytes(&x_2d)
            .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp extract x bytes: {e:?}")))?;

        // hidden = x @ A.T. Transposed kernel against the A buffer
        // (treated as [n_dim=rank, k_dim=in_features]). The `_bytes`
        // variant takes raw f32 bytes and returns raw f32 bytes shaped
        // logically as `[row_count, 1, rank]` — we keep the result as
        // bytes so the next dispatch can consume it without a candle
        // round trip.
        let hidden_bytes = kernels::dispatch_linear_decode_cached_bf16_weights_transposed_bytes(
            self.vk_device.as_ref(),
            &x_2d_bytes,
            self.a_buffer.as_ref(),
            row_count,
            self.in_features,
            self.rank,
        )
        .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp x@A.T: {e:?}")))?;

        // delta_unscaled = hidden @ B.T. Same kernel against B buffer.
        // The `hidden_bytes` buffer is already row-major `[row_count,
        // rank]` (the trailing `1` in the logical shape is a no-op for
        // contiguous layouts), so we pass it straight in.
        let delta_bytes = kernels::dispatch_linear_decode_cached_bf16_weights_transposed_bytes(
            self.vk_device.as_ref(),
            &hidden_bytes,
            self.b_buffer.as_ref(),
            row_count,
            self.rank,
            self.out_features,
        )
        .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp hidden@B.T: {e:?}")))?;
        let mut out_dims: Vec<usize> = x_dims[..x_dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        // Materialize the final f32 result back into a candle candle_core::Tensor so
        // the rest of the op (scale, dtype cast, storage extraction)
        // can keep using candle ops unchanged.
        let delta_unscaled = tensor_from_f32_bytes(
            &delta_bytes,
            out_dims.as_slice(),
            candle_core::DType::F32,
        )
        .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp build delta tensor: {e:?}")))?;
        let delta_scaled = (delta_unscaled * self.scale as f64)
            .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp scale: {e:?}")))?;
        let delta_typed = if delta_scaled.dtype() == self.out_dtype {
            delta_scaled
        } else {
            delta_scaled
                .to_dtype(self.out_dtype)
                .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp out cast: {e:?}")))?
        };

        let storage = delta_typed
            .storage_and_layout()
            .0
            .try_clone(delta_typed.layout())
            .map_err(|e| {
                candle_core::Error::Msg(format!("VulkanLoraOp out storage clone: {e:?}"))
            })?;
        let cpu_storage = match storage {
            candle_core::Storage::Cpu(s) => s,
            _ => {
                return Err(candle_core::Error::Msg(
                    "VulkanLoraOp: expected CPU storage from kernel result".into(),
                ));
            }
        };
        Ok((cpu_storage, candle_core::Shape::from(out_dims.as_slice())))
    }

    fn bwd(
        &self,
        x: &candle_core::Tensor,
        a: &candle_core::Tensor,
        b: &candle_core::Tensor,
        _res: &candle_core::Tensor,
        grad_y: &candle_core::Tensor,
    ) -> candle_core::Result<(Option<candle_core::Tensor>, Option<candle_core::Tensor>, Option<candle_core::Tensor>)> {
        // Read A and B values straight from the registry buffers (the
        // canonical source of truth post-Phase 4.x — the candle CPU
        // storage of these Vars is lazily synced and may be stale
        // between training steps under the on-device optimizer path).
        // x is an upstream activation, not a LoRA Var, so we still
        // read it from candle CPU storage.
        let scale = self.scale as f64;
        let x_f32 = if x.dtype() == candle_core::DType::F32 {
            x.clone()
        } else {
            x.to_dtype(candle_core::DType::F32)?
        };
        let a_f32 = buffer_to_tensor_inline(
            self.vk_device.as_ref(),
            self.a_buffer.as_ref(),
            &[self.rank, self.in_features],
            a.dtype(),
        )
        .map_err(|e| {
            candle_core::Error::Msg(format!("VulkanLoraOp::bwd buffer_to_tensor A: {e:?}"))
        })?
        .to_dtype(candle_core::DType::F32)?;
        let b_f32 = buffer_to_tensor_inline(
            self.vk_device.as_ref(),
            self.b_buffer.as_ref(),
            &[self.out_features, self.rank],
            b.dtype(),
        )
        .map_err(|e| {
            candle_core::Error::Msg(format!("VulkanLoraOp::bwd buffer_to_tensor B: {e:?}"))
        })?
        .to_dtype(candle_core::DType::F32)?;
        let grad_y_f32 = if grad_y.dtype() == candle_core::DType::F32 {
            grad_y.clone()
        } else {
            grad_y.to_dtype(candle_core::DType::F32)?
        };

        let grad_d = (grad_y_f32 * scale)?;
        // Recompute h = x @ A.T (cheap: small rank dim).
        let h = x_f32.broadcast_matmul(&a_f32.t()?)?;
        // grad_h = grad_d @ B, B is [out, rank] so this is matmul not
        // transpose-matmul.
        let grad_h = grad_d.broadcast_matmul(&b_f32)?;
        // grad_x = grad_h @ A
        let grad_x_f32 = grad_h.broadcast_matmul(&a_f32)?;

        // Collapse leading batch dims for the outer-product gradients.
        let last = x.dims().len() - 1;
        let total_batch: usize = x.dims()[..last].iter().product();
        let in_features = x.dims()[last];
        let grad_h_2d = grad_h.reshape((total_batch, self.rank))?;
        let x_2d = x_f32.reshape((total_batch, in_features))?;
        // grad_A[r, k] = sum_b grad_h[b, r] * x[b, k] = (grad_h.T @ x)
        let grad_a_f32 = grad_h_2d.t()?.contiguous()?.broadcast_matmul(&x_2d)?;

        let grad_d_2d = grad_d.reshape((total_batch, self.out_features))?;
        let h_2d = h.reshape((total_batch, self.rank))?;
        // grad_B[o, r] = sum_b grad_d[b, o] * h[b, r] = (grad_d.T @ h)
        let grad_b_f32 = grad_d_2d.t()?.contiguous()?.broadcast_matmul(&h_2d)?;

        let grad_x = grad_x_f32.to_dtype(x.dtype())?;
        let grad_a = grad_a_f32.to_dtype(a.dtype())?;
        let grad_b = grad_b_f32.to_dtype(b.dtype())?;
        Ok((Some(grad_x), Some(grad_a), Some(grad_b)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic forward parity test: x.apply_op3(VulkanLoraOp{...})
    /// must match `(x @ A.T @ B.T) * scale` to bf16 precision.
    #[test]
    fn vulkan_lora_op_forward_parity_small() -> anyhow::Result<()> {
        let Ok(vk_device) = VulkanDevice::new() else {
            eprintln!("skipping: no Vulkan device");
            return Ok(());
        };
        let vk_device = Arc::new(vk_device);
        let device = candle_core::Device::Cpu;
        let t = 4usize;
        let in_features = 8usize;
        let rank = 4usize;
        let out_features = 6usize;
        let scale = 0.5f32;

        let x_data: Vec<f32> = (0..t * in_features).map(|i| (i as f32) * 0.013).collect();
        let a_data: Vec<f32> = (0..rank * in_features)
            .map(|i| (i as f32) * 0.017)
            .collect();
        let b_data: Vec<f32> = (0..out_features * rank)
            .map(|i| (i as f32) * 0.011)
            .collect();
        let x = candle_core::Tensor::from_vec(x_data, (1, t, in_features), &device)?.to_dtype(candle_core::DType::BF16)?;
        let a = candle_core::Tensor::from_vec(a_data, (rank, in_features), &device)?.to_dtype(candle_core::DType::BF16)?;
        let b = candle_core::Tensor::from_vec(b_data, (out_features, rank), &device)?.to_dtype(candle_core::DType::BF16)?;

        // CPU baseline (manual F32; candle CPU doesn't support BF16
        // matmul).
        let x_f = x.to_dtype(candle_core::DType::F32)?;
        let a_f = a.to_dtype(candle_core::DType::F32)?;
        let b_f = b.to_dtype(candle_core::DType::F32)?;
        let cpu_delta = (x_f
            .broadcast_matmul(&a_f.t()?)?
            .broadcast_matmul(&b_f.t()?)?
            * scale as f64)?
            .to_dtype(candle_core::DType::BF16)?;

        // Vulkan path — upload A and B as registry-resident.
        let a_buf = Arc::new(upload_tensor_bf16_packed_buffer_inline(
            vk_device.as_ref(),
            &a,
        )?);
        let b_buf = Arc::new(upload_tensor_bf16_packed_buffer_inline(
            vk_device.as_ref(),
            &b,
        )?);
        let op = VulkanLoraOp {
            vk_device: vk_device.clone(),
            a_buffer: a_buf,
            b_buffer: b_buf,
            rank,
            in_features,
            out_features,
            scale,
            out_dtype: candle_core::DType::BF16,
        };
        let vk_delta = x.apply_op3(&a, &b, op)?;

        assert_eq!(vk_delta.dims(), cpu_delta.dims());
        assert_eq!(vk_delta.dtype(), cpu_delta.dtype());
        let cpu_v: Vec<f32> = cpu_delta.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
        let vk_v: Vec<f32> = vk_delta.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
        for (i, (c, v)) in cpu_v.iter().zip(vk_v.iter()).enumerate() {
            let abs = (c - v).abs();
            let rel = abs / c.abs().max(1e-3);
            assert!(
                abs < 5e-2 || rel < 5e-2,
                "idx {i}: cpu={c:.6} vk={v:.6} abs={abs:e} rel={rel:e}"
            );
        }
        Ok(())
    }

    /// Backward parity test: train against a synthetic loss
    /// `loss = sum(out * grad_y)` and verify gradients on A and B
    /// match the candle-CPU autograd reference.
    #[test]
    fn vulkan_lora_op_backward_parity_small() -> anyhow::Result<()> {
        let Ok(vk_device) = VulkanDevice::new() else {
            eprintln!("skipping: no Vulkan device");
            return Ok(());
        };
        let vk_device = Arc::new(vk_device);
        let device = candle_core::Device::Cpu;
        let t = 5usize;
        let in_features = 8usize;
        let rank = 4usize;
        let out_features = 6usize;
        let scale = 0.7f32;

        let x_data: Vec<f32> = (0..t * in_features).map(|i| (i as f32) * 0.011).collect();
        let a_data: Vec<f32> = (0..rank * in_features)
            .map(|i| (i as f32) * 0.013)
            .collect();
        let b_data: Vec<f32> = (0..out_features * rank)
            .map(|i| (i as f32) * 0.017)
            .collect();

        let x = candle_core::Tensor::from_vec(x_data, (1, t, in_features), &device)?.to_dtype(candle_core::DType::BF16)?;
        let a_var = candle_core::Var::from_tensor(
            &candle_core::Tensor::from_vec(a_data, (rank, in_features), &device)?.to_dtype(candle_core::DType::BF16)?,
        )?;
        let b_var = candle_core::Var::from_tensor(
            &candle_core::Tensor::from_vec(b_data, (out_features, rank), &device)?.to_dtype(candle_core::DType::BF16)?,
        )?;

        // Synthetic upstream gradient — a tensor that we'll multiply
        // by the output before sum-reduction, so dL/d_out = grad_y.
        let grad_y_data: Vec<f32> = (0..t * out_features)
            .map(|i| ((i as i32 - 10) as f32) * 0.019)
            .collect();
        let grad_y =
            candle_core::Tensor::from_vec(grad_y_data, (1, t, out_features), &device)?.to_dtype(candle_core::DType::F32)?;

        // ---- Vulkan path: construct VulkanLoraOp + apply_op3,
        // backward, extract grads.
        let a_buf = Arc::new(upload_tensor_bf16_packed_buffer_inline(
            vk_device.as_ref(),
            a_var.as_tensor(),
        )?);
        let b_buf = Arc::new(upload_tensor_bf16_packed_buffer_inline(
            vk_device.as_ref(),
            b_var.as_tensor(),
        )?);
        let op = VulkanLoraOp {
            vk_device: vk_device.clone(),
            a_buffer: a_buf,
            b_buffer: b_buf,
            rank,
            in_features,
            out_features,
            scale,
            out_dtype: candle_core::DType::BF16,
        };
        let vk_out = x.apply_op3(a_var.as_tensor(), b_var.as_tensor(), op)?;
        let vk_loss = (vk_out.to_dtype(candle_core::DType::F32)? * &grad_y)?.sum_all()?;
        let vk_grads = vk_loss.backward()?;
        let vk_grad_a = vk_grads
            .get(a_var.as_tensor())
            .expect("Vulkan grad_A present")
            .clone();
        let vk_grad_b = vk_grads
            .get(b_var.as_tensor())
            .expect("Vulkan grad_B present")
            .clone();

        // ---- CPU baseline using the same Vars but going through the
        // candle CPU broadcast_matmul chain (the existing
        // `compute_lora_delta`).
        let cpu_proj = crate::lora_loader::LoraProjectionWeights {
            a: a_var.as_tensor().clone(),
            b: b_var.as_tensor().clone(),
        };
        // Promote x to f32 for CPU matmul (BF16 not supported on CPU).
        let x_for_cpu = x.to_dtype(candle_core::DType::F32)?;
        let a_cpu = a_var.as_tensor().to_dtype(candle_core::DType::F32)?;
        let b_cpu = b_var.as_tensor().to_dtype(candle_core::DType::F32)?;
        let cpu_out_unscaled = x_for_cpu
            .broadcast_matmul(&a_cpu.t()?)?
            .broadcast_matmul(&b_cpu.t()?)?;
        let cpu_out = (cpu_out_unscaled * scale as f64)?;
        let cpu_loss = (cpu_out * &grad_y)?.sum_all()?;
        let cpu_grads = cpu_loss.backward()?;
        let cpu_grad_a = cpu_grads
            .get(a_var.as_tensor())
            .expect("CPU grad_A present")
            .clone();
        let cpu_grad_b = cpu_grads
            .get(b_var.as_tensor())
            .expect("CPU grad_B present")
            .clone();
        // Silence unused warning for the proj struct (kept to make
        // the test cross-check the actual production tensor type).
        let _ = cpu_proj;

        // Compare grad_A.
        assert_eq!(vk_grad_a.dims(), cpu_grad_a.dims());
        let vk_a_v: Vec<f32> = vk_grad_a.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
        let cpu_a_v: Vec<f32> = cpu_grad_a.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
        for (i, (v, c)) in vk_a_v.iter().zip(cpu_a_v.iter()).enumerate() {
            let abs = (v - c).abs();
            let rel = abs / c.abs().max(1e-3);
            assert!(
                abs < 5e-2 || rel < 5e-2,
                "grad_A[{i}]: vk={v:.6} cpu={c:.6} abs={abs:e} rel={rel:e}"
            );
        }
        // Compare grad_B.
        assert_eq!(vk_grad_b.dims(), cpu_grad_b.dims());
        let vk_b_v: Vec<f32> = vk_grad_b.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
        let cpu_b_v: Vec<f32> = cpu_grad_b.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
        for (i, (v, c)) in vk_b_v.iter().zip(cpu_b_v.iter()).enumerate() {
            let abs = (v - c).abs();
            let rel = abs / c.abs().max(1e-3);
            assert!(
                abs < 5e-2 || rel < 5e-2,
                "grad_B[{i}]: vk={v:.6} cpu={c:.6} abs={abs:e} rel={rel:e}"
            );
        }
        Ok(())
    }
}
