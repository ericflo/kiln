//! `CustomOp3` wrapper for Vulkan-dispatched LoRA delta application.
//!
//! Phase 4.1 of the residency plan. Implements `CustomOp3` so the
//! on-device LoRA delta dispatch carries an analytic backward that
//! returns gradients for x, A, and B — making it safe for both
//! training-time and inference-time forward passes. (An earlier
//! version returned a candle-leaf Tensor with no autograd back-link
//! and had to be gated to inference-only; this CustomOp3 wrapper
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

use anyhow::Context as _;
use candle_core::backend::BackendStorage;
use candle_core::op::BackpropOp;
use candle_core::{CpuStorage, CustomOp3, DType, Layout, Shape, Storage, Tensor};

use kiln_vulkan_kernel::{VulkanBuffer, VulkanDevice, kernels};

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
    pub out_dtype: DType,
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

impl CustomOp3 for VulkanLoraOp {
    fn name(&self) -> &'static str {
        "kiln-vulkan-lora-delta"
    }

    fn cpu_fwd(
        &self,
        s_x: &CpuStorage,
        l_x: &Layout,
        _s_a: &CpuStorage,
        _l_a: &Layout,
        _s_b: &CpuStorage,
        _l_b: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
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
            let zero = Tensor::zeros(out_dims.as_slice(), self.out_dtype, &candle_core::Device::Cpu)
                .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp empty zeros: {e:?}")))?;
            let storage = zero
                .storage_and_layout()
                .0
                .try_clone(zero.layout())
                .map_err(|e| {
                    candle_core::Error::Msg(format!("VulkanLoraOp empty storage clone: {e:?}"))
                })?;
            let cpu_storage = match storage {
                Storage::Cpu(s) => s,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "VulkanLoraOp: expected CPU storage from empty zeros".into(),
                    ));
                }
            };
            return Ok((cpu_storage, Shape::from(out_dims.as_slice())));
        }

        // Wrap the x storage in a Tensor so we can call the kernel
        // dispatch helpers (which expect &Tensor).
        let storage = Storage::Cpu(s_x.clone());
        let x_tensor = Tensor::from_storage(
            storage,
            Shape::from(x_dims.to_vec()),
            BackpropOp::none(),
            false,
        );
        let x_f32 = if x_tensor.dtype() == DType::F32 {
            x_tensor
        } else {
            x_tensor
                .to_dtype(DType::F32)
                .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp x→f32: {e:?}")))?
        };
        let x_2d = x_f32
            .reshape((row_count, self.in_features))
            .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp reshape x: {e:?}")))?
            .contiguous()
            .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp x contiguous: {e:?}")))?;

        // hidden = x @ A.T. Transposed kernel against the A buffer
        // (treated as [n_dim=rank, k_dim=in_features]).
        let hidden = kernels::dispatch_linear_decode_cached_bf16_weights_transposed(
            self.vk_device.as_ref(),
            &x_2d,
            self.a_buffer.as_ref(),
            row_count,
            self.in_features,
            self.rank,
        )
        .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp x@A.T: {e:?}")))?;
        let hidden_2d = hidden
            .reshape((row_count, self.rank))
            .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp reshape hidden: {e:?}")))?
            .contiguous()
            .map_err(|e| {
                candle_core::Error::Msg(format!("VulkanLoraOp hidden contiguous: {e:?}"))
            })?;

        // delta_unscaled = hidden @ B.T. Same kernel against B buffer.
        let delta_unscaled = kernels::dispatch_linear_decode_cached_bf16_weights_transposed(
            self.vk_device.as_ref(),
            &hidden_2d,
            self.b_buffer.as_ref(),
            row_count,
            self.rank,
            self.out_features,
        )
        .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp hidden@B.T: {e:?}")))?;
        let mut out_dims: Vec<usize> = x_dims[..x_dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        let delta_unscaled = delta_unscaled
            .reshape(out_dims.as_slice())
            .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp reshape out: {e:?}")))?;
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
            Storage::Cpu(s) => s,
            _ => {
                return Err(candle_core::Error::Msg(
                    "VulkanLoraOp: expected CPU storage from kernel result".into(),
                ));
            }
        };
        Ok((cpu_storage, Shape::from(out_dims.as_slice())))
    }

    fn bwd(
        &self,
        x: &Tensor,
        a: &Tensor,
        b: &Tensor,
        _res: &Tensor,
        grad_y: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        // Read A and B values straight from the registry buffers (the
        // canonical source of truth post-Phase 4.x — the candle CPU
        // storage of these Vars is lazily synced and may be stale
        // between training steps under the on-device optimizer path).
        // x is an upstream activation, not a LoRA Var, so we still
        // read it from candle CPU storage.
        let scale = self.scale as f64;
        let x_f32 = if x.dtype() == DType::F32 { x.clone() } else { x.to_dtype(DType::F32)? };
        let a_f32 = kernels::buffer_to_tensor(
            self.vk_device.as_ref(),
            self.a_buffer.as_ref(),
            &[self.rank, self.in_features],
            a.dtype(),
        )
        .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp::bwd buffer_to_tensor A: {e:?}")))?
        .to_dtype(DType::F32)?;
        let b_f32 = kernels::buffer_to_tensor(
            self.vk_device.as_ref(),
            self.b_buffer.as_ref(),
            &[self.out_features, self.rank],
            b.dtype(),
        )
        .map_err(|e| candle_core::Error::Msg(format!("VulkanLoraOp::bwd buffer_to_tensor B: {e:?}")))?
        .to_dtype(DType::F32)?;
        let grad_y_f32 = if grad_y.dtype() == DType::F32 {
            grad_y.clone()
        } else {
            grad_y.to_dtype(DType::F32)?
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
    use candle_core::Device;

    /// Synthetic forward parity test: x.apply_op3(VulkanLoraOp{...})
    /// must match `(x @ A.T @ B.T) * scale` to bf16 precision.
    #[test]
    fn vulkan_lora_op_forward_parity_small() -> anyhow::Result<()> {
        let Ok(vk_device) = VulkanDevice::new() else {
            eprintln!("skipping: no Vulkan device");
            return Ok(());
        };
        let vk_device = Arc::new(vk_device);
        let device = Device::Cpu;
        let t = 4usize;
        let in_features = 8usize;
        let rank = 4usize;
        let out_features = 6usize;
        let scale = 0.5f32;

        let x_data: Vec<f32> = (0..t * in_features).map(|i| (i as f32) * 0.013).collect();
        let a_data: Vec<f32> = (0..rank * in_features).map(|i| (i as f32) * 0.017).collect();
        let b_data: Vec<f32> = (0..out_features * rank).map(|i| (i as f32) * 0.011).collect();
        let x = Tensor::from_vec(x_data, (1, t, in_features), &device)?
            .to_dtype(DType::BF16)?;
        let a = Tensor::from_vec(a_data, (rank, in_features), &device)?
            .to_dtype(DType::BF16)?;
        let b = Tensor::from_vec(b_data, (out_features, rank), &device)?
            .to_dtype(DType::BF16)?;

        // CPU baseline (manual F32; candle CPU doesn't support BF16
        // matmul).
        let x_f = x.to_dtype(DType::F32)?;
        let a_f = a.to_dtype(DType::F32)?;
        let b_f = b.to_dtype(DType::F32)?;
        let cpu_delta = (x_f
            .broadcast_matmul(&a_f.t()?)?
            .broadcast_matmul(&b_f.t()?)?
            * scale as f64)?
            .to_dtype(DType::BF16)?;

        // Vulkan path — upload A and B as registry-resident.
        let a_buf =
            Arc::new(kernels::upload_tensor_bf16_packed_buffer(vk_device.as_ref(), &a)?);
        let b_buf =
            Arc::new(kernels::upload_tensor_bf16_packed_buffer(vk_device.as_ref(), &b)?);
        let op = VulkanLoraOp {
            vk_device: vk_device.clone(),
            a_buffer: a_buf,
            b_buffer: b_buf,
            rank,
            in_features,
            out_features,
            scale,
            out_dtype: DType::BF16,
        };
        let vk_delta = x.apply_op3(&a, &b, op)?;

        assert_eq!(vk_delta.dims(), cpu_delta.dims());
        assert_eq!(vk_delta.dtype(), cpu_delta.dtype());
        let cpu_v: Vec<f32> = cpu_delta.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let vk_v: Vec<f32> = vk_delta.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
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
        let device = Device::Cpu;
        let t = 5usize;
        let in_features = 8usize;
        let rank = 4usize;
        let out_features = 6usize;
        let scale = 0.7f32;

        let x_data: Vec<f32> = (0..t * in_features).map(|i| (i as f32) * 0.011).collect();
        let a_data: Vec<f32> = (0..rank * in_features).map(|i| (i as f32) * 0.013).collect();
        let b_data: Vec<f32> = (0..out_features * rank).map(|i| (i as f32) * 0.017).collect();

        let x = Tensor::from_vec(x_data, (1, t, in_features), &device)?
            .to_dtype(DType::BF16)?;
        let a_var = candle_core::Var::from_tensor(
            &Tensor::from_vec(a_data, (rank, in_features), &device)?
                .to_dtype(DType::BF16)?,
        )?;
        let b_var = candle_core::Var::from_tensor(
            &Tensor::from_vec(b_data, (out_features, rank), &device)?
                .to_dtype(DType::BF16)?,
        )?;

        // Synthetic upstream gradient — a tensor that we'll multiply
        // by the output before sum-reduction, so dL/d_out = grad_y.
        let grad_y_data: Vec<f32> = (0..t * out_features)
            .map(|i| ((i as i32 - 10) as f32) * 0.019)
            .collect();
        let grad_y = Tensor::from_vec(grad_y_data, (1, t, out_features), &device)?
            .to_dtype(DType::F32)?;

        // ---- Vulkan path: construct VulkanLoraOp + apply_op3,
        // backward, extract grads.
        let a_buf = Arc::new(kernels::upload_tensor_bf16_packed_buffer(
            vk_device.as_ref(),
            a_var.as_tensor(),
        )?);
        let b_buf = Arc::new(kernels::upload_tensor_bf16_packed_buffer(
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
            out_dtype: DType::BF16,
        };
        let vk_out = x.apply_op3(a_var.as_tensor(), b_var.as_tensor(), op)?;
        let vk_loss = (vk_out.to_dtype(DType::F32)? * &grad_y)?.sum_all()?;
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
        let x_for_cpu = x.to_dtype(DType::F32)?;
        let a_cpu = a_var.as_tensor().to_dtype(DType::F32)?;
        let b_cpu = b_var.as_tensor().to_dtype(DType::F32)?;
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
        let vk_a_v: Vec<f32> = vk_grad_a.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let cpu_a_v: Vec<f32> = cpu_grad_a.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
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
        let vk_b_v: Vec<f32> = vk_grad_b.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let cpu_b_v: Vec<f32> = cpu_grad_b.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
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
