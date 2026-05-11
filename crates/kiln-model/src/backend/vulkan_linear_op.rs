//! `CustomOp1` wrapper for Vulkan-dispatched linear (matmul) projections.
//!
//! Phase 2 sub-step 1 of the residency plan. The training forward path is
//! dominated by `[B, T, hidden] @ [hidden, out_dim] -> [B, T, out_dim]`
//! matmuls in q/k/v/o/gate/up/down projections; on Vulkan today they fall
//! through to the candle CPU bf16 path because `Tensor::from_vec` (the
//! result type of the existing `dispatch_linear_decode_*` functions)
//! produces an autograd leaf — wiring the dispatch directly into the
//! forward pass would silently break `loss.backward()`.
//!
//! This module wraps the Vulkan dispatch in a [`candle_core::CustomOp1`]
//! that:
//! - Captures the (already-uploaded) [`kiln_vulkan_kernel::VulkanBuffer`]
//!   for the transposed weight as op state, since the weight is frozen
//!   during LoRA training and never accumulates gradients.
//! - Implements `cpu_fwd` by uploading `x`, dispatching the existing
//!   `linear_decode_batched` kernel (which already supports any batch
//!   size — it's mis-named "decode"), and returning the f32 result.
//! - Implements `bwd` analytically: `dX = grad_y @ W` where
//!   `W = weight_t.t()`. The backward is also a matmul, dispatchable
//!   through the same kernel (with the transposed-of-transposed weight),
//!   though the first cut routes it through candle CPU for parity safety.
//!
//! The wrapper is **not yet wired into `forward.rs`** — that's a follow-up
//! that needs end-to-end parity testing on the live training path. This
//! module ships the building block plus a synthetic parity test so the
//! integration step has a known-good reference.

use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::backend::BackendStorage;
use candle_core::op::BackpropOp;
use candle_core::{CpuStorage, CustomOp1, DType, Device, Layout, Shape, Storage, Tensor};

use kiln_vulkan_kernel::{VulkanBuffer, VulkanDevice, kernels};

/// Marker for which Vulkan kernel variant to dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLayout {
    /// Weight buffer is f32 in row-major `[hidden, out_dim]` order.
    F32,
    /// Weight buffer is the bf16-packed layout produced by
    /// [`kernels::upload_tensor_bf16_packed_buffer`] — two bf16 lanes
    /// per u32 in row-major `[hidden, out_dim]` order. Shaders expand
    /// each lane via `uintBitsToFloat(bits << 16)`.
    Bf16Packed,
}

/// Op state for [`VulkanLinearOp`]. Captures everything needed to
/// dispatch the matmul without holding a `&self` reference to a
/// `VulkanBackend` (which would not satisfy `'static`).
pub struct VulkanLinearOp {
    pub vk_device: Arc<VulkanDevice>,
    pub weight_buffer: Arc<VulkanBuffer>,
    pub weight_layout: WeightLayout,
    pub hidden: usize,
    pub out_dim: usize,
    /// Output dtype for the final tensor. Lets the wrapper be the single
    /// place that decides whether to keep the f32 result or cast back to
    /// the input dtype, avoiding a second `.to_dtype()` round-trip in
    /// every call site.
    pub out_dtype: DType,
    /// The candle weight tensor backing `weight_buffer`. Held here so
    /// `bwd` can compute `dX = grad_y @ W` without taking a second copy
    /// of the device buffer back to CPU (the candle tensor's CPU storage
    /// is already there). For frozen LoRA-base weights this is the same
    /// `Arc<Storage>` the rest of the model holds, so capturing the
    /// `Tensor` here adds no real memory cost.
    pub weight_t: Tensor,
}

impl std::fmt::Debug for VulkanLinearOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanLinearOp")
            .field("hidden", &self.hidden)
            .field("out_dim", &self.out_dim)
            .field("weight_layout", &self.weight_layout)
            .field("out_dtype", &self.out_dtype)
            .finish()
    }
}

impl CustomOp1 for VulkanLinearOp {
    fn name(&self) -> &'static str {
        "kiln-vulkan-linear"
    }

    fn cpu_fwd(
        &self,
        s_x: &CpuStorage,
        l_x: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let dims = l_x.shape().dims();
        if dims.is_empty() {
            return Err(candle_core::Error::Msg(
                "VulkanLinearOp: x must have at least one dim".into(),
            ));
        }
        let row_count: usize = dims[..dims.len() - 1].iter().product();
        let inner = dims[dims.len() - 1];
        if inner != self.hidden {
            return Err(candle_core::Error::Msg(format!(
                "VulkanLinearOp: x last dim {inner} != hidden {}",
                self.hidden
            )));
        }

        // Promote x to f32 — the existing kernel takes f32 input. This
        // round-trip is cheap relative to the matmul itself (e.g.
        // T=1500 H=2560 ~ 7.7 MB to convert vs 70 GFLOP to compute).
        // Wrapping the storage in a Tensor lets us reuse the existing
        // dispatch path which expects `&Tensor`.
        let storage = Storage::Cpu(s_x.clone());
        let x_tensor = Tensor::from_storage(
            storage,
            Shape::from(l_x.shape().dims()),
            BackpropOp::none(),
            false,
        );
        let x_f32 = if x_tensor.dtype() == DType::F32 {
            x_tensor
        } else {
            x_tensor
                .to_dtype(DType::F32)
                .map_err(|e| candle_core::Error::Msg(format!("VulkanLinearOp x→f32: {e:?}")))?
        };
        // The kernel expects shape `[N, 1, hidden]` for the batched path —
        // any leading non-trivial batch/seq layout collapses to
        // `[row_count, 1, hidden]` with the same memory order.
        let dispatch_x = if dims.len() == 3 && dims[1] == 1 {
            x_f32
        } else {
            x_f32
                .reshape((row_count, 1usize, self.hidden))
                .map_err(|e| {
                    candle_core::Error::Msg(format!("VulkanLinearOp reshape x: {e:?}"))
                })?
        };

        let out_tensor = match self.weight_layout {
            WeightLayout::F32 => kernels::dispatch_linear_decode_cached(
                self.vk_device.as_ref(),
                &dispatch_x,
                self.weight_buffer.as_ref(),
                row_count,
                self.hidden,
                self.out_dim,
            ),
            WeightLayout::Bf16Packed => kernels::dispatch_linear_decode_cached_bf16_weights(
                self.vk_device.as_ref(),
                &dispatch_x,
                self.weight_buffer.as_ref(),
                row_count,
                self.hidden,
                self.out_dim,
            ),
        }
        .map_err(|e| candle_core::Error::Msg(format!("VulkanLinearOp dispatch: {e:?}")))?;

        // Restore the original leading dims with `out_dim` swapped in for
        // `inner`. CustomOp1's output Shape is what candle uses to size
        // downstream ops, so it has to match what a `broadcast_matmul`
        // would have produced.
        let mut out_dims: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_dim);
        let out_tensor = out_tensor.reshape(out_dims.as_slice()).map_err(|e| {
            candle_core::Error::Msg(format!("VulkanLinearOp reshape out: {e:?}"))
        })?;

        // Cast to the requested output dtype. F32 → BF16 is a tight loop
        // and trivially small relative to the matmul.
        let out_tensor = if out_tensor.dtype() == self.out_dtype {
            out_tensor
        } else {
            out_tensor.to_dtype(self.out_dtype).map_err(|e| {
                candle_core::Error::Msg(format!("VulkanLinearOp cast out dtype: {e:?}"))
            })?
        };

        let storage = out_tensor
            .storage_and_layout()
            .0
            .try_clone(out_tensor.layout())
            .map_err(|e| {
                candle_core::Error::Msg(format!("VulkanLinearOp out storage clone: {e:?}"))
            })?;
        let cpu_storage = match storage {
            Storage::Cpu(s) => s,
            _ => {
                return Err(candle_core::Error::Msg(
                    "VulkanLinearOp: expected CPU storage from kernel result".into(),
                ));
            }
        };

        Ok((cpu_storage, Shape::from(out_dims.as_slice())))
    }

    fn bwd(
        &self,
        _x: &Tensor,
        _y: &Tensor,
        grad_y: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        // y = x @ weight_t        — weight_t has shape [hidden, out_dim]
        // dX = grad_y @ weight_t.T — i.e. grad_y has shape [..., out_dim],
        //                            weight_t.T has shape [out_dim, hidden],
        //                            dX has shape [..., hidden].
        //
        // Try the Vulkan transposed-weight kernel first: it dispatches
        // against the SAME bf16-packed buffer the forward used (no
        // re-upload of a transposed view). Falls back to candle CPU
        // broadcast_matmul if the kernel preconditions don't hold (e.g.
        // weight is f32, or the kernel returns None for shape reasons).
        if self.weight_layout == WeightLayout::Bf16Packed {
            let dims = grad_y.shape().dims().to_vec();
            let row_count: usize = dims[..dims.len() - 1].iter().product();
            let grad_y_f32 = if grad_y.dtype() == DType::F32 {
                grad_y.clone()
            } else {
                grad_y
                    .to_dtype(DType::F32)
                    .map_err(|e| candle_core::Error::Msg(format!("bwd grad_y→f32: {e:?}")))?
            };
            let dispatch_x = grad_y_f32
                .reshape((row_count, self.out_dim))
                .map_err(|e| {
                    candle_core::Error::Msg(format!("bwd reshape grad_y: {e:?}"))
                })?;
            let dx_3d = kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_transposed(
                self.vk_device.as_ref(),
                &dispatch_x,
                self.weight_buffer.as_ref(),
                row_count,
                self.out_dim,
                self.hidden,
            )
            .map_err(|e| candle_core::Error::Msg(format!("bwd transposed dispatch: {e:?}")))?;
            // dx_3d is [row_count, 1, hidden]; restore caller's leading dims.
            let mut out_dims: Vec<usize> = dims[..dims.len() - 1].to_vec();
            out_dims.push(self.hidden);
            let dx_f32 = dx_3d.reshape(out_dims.as_slice()).map_err(|e| {
                candle_core::Error::Msg(format!("bwd reshape dx: {e:?}"))
            })?;
            let dx = if self.out_dtype == DType::F32 {
                dx_f32
            } else {
                dx_f32.to_dtype(self.out_dtype).map_err(|e| {
                    candle_core::Error::Msg(format!("bwd cast dx: {e:?}"))
                })?
            };
            return Ok(Some(dx));
        }

        // F32 weight path: fall back to candle CPU broadcast_matmul.
        let weight_t = if self.weight_t.dtype() == DType::F32 {
            self.weight_t.clone()
        } else {
            self.weight_t
                .to_dtype(DType::F32)
                .map_err(|e| candle_core::Error::Msg(format!("bwd weight→f32: {e:?}")))?
        };
        let weight = weight_t
            .transpose(0, 1)
            .map_err(|e| candle_core::Error::Msg(format!("bwd weight transpose: {e:?}")))?
            .contiguous()
            .map_err(|e| candle_core::Error::Msg(format!("bwd weight contiguous: {e:?}")))?;
        let grad_y_f32 = if grad_y.dtype() == DType::F32 {
            grad_y.clone()
        } else {
            grad_y
                .to_dtype(DType::F32)
                .map_err(|e| candle_core::Error::Msg(format!("bwd grad_y→f32: {e:?}")))?
        };
        let dx_f32 = grad_y_f32
            .broadcast_matmul(&weight)
            .map_err(|e| candle_core::Error::Msg(format!("bwd matmul: {e:?}")))?;
        // Match the input's dtype on the gradient — candle's autograd
        // expects dX.dtype() == X.dtype(). Since the caller's X may have
        // been bf16, return bf16 to avoid a downstream cast surprise.
        // The dtype to return is `out_dtype` of THIS op which mirrors X
        // dtype by convention.
        let dx = if self.out_dtype == DType::F32 {
            dx_f32
        } else {
            dx_f32
                .to_dtype(self.out_dtype)
                .map_err(|e| candle_core::Error::Msg(format!("bwd cast dx: {e:?}")))?
        };
        Ok(Some(dx))
    }
}

/// Convenience constructor that uploads (or reuses a cached upload of) the
/// weight buffer and returns a ready-to-apply [`VulkanLinearOp`].
///
/// `weight_t` is the row-major `[hidden, out_dim]` transposed weight. The
/// caller is expected to have computed it once at load time. The same
/// candle `Tensor` is captured into op state so `bwd` can compute
/// `dX = grad_y @ weight_t.T` without re-downloading the weight.
pub fn build_op(
    vk_device: Arc<VulkanDevice>,
    weight_buffer: Arc<VulkanBuffer>,
    weight_t: Tensor,
    weight_layout: WeightLayout,
    hidden: usize,
    out_dim: usize,
    out_dtype: DType,
) -> VulkanLinearOp {
    VulkanLinearOp {
        vk_device,
        weight_buffer,
        weight_layout,
        hidden,
        out_dim,
        out_dtype,
        weight_t,
    }
}

/// Run a Vulkan-dispatched matmul over a borrowed weight tensor without
/// going through `apply_op1` — used by parity tests and by inference
/// paths that don't need autograd.
///
/// The autograd-safe entry is `tensor.apply_op1(VulkanLinearOp { ... })`,
/// but `apply_op1` requires `bwd` to be implemented. Until that lands,
/// callers that only need the forward result use this helper directly.
pub fn dispatch_forward_only(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight_buffer: &VulkanBuffer,
    weight_layout: WeightLayout,
    hidden: usize,
    out_dim: usize,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let row_count: usize = dims[..dims.len() - 1].iter().product();
    let x_f32 = if x.dtype() == DType::F32 {
        x.clone()
    } else {
        x.to_dtype(DType::F32)
            .context("dispatch_forward_only: x→f32")?
    };
    let dispatch_x = if dims.len() == 3 && dims[1] == 1 {
        x_f32
    } else {
        x_f32
            .reshape((row_count, 1usize, hidden))
            .context("dispatch_forward_only: reshape x")?
    };
    let out = match weight_layout {
        WeightLayout::F32 => kernels::dispatch_linear_decode_cached(
            vk_device,
            &dispatch_x,
            weight_buffer,
            row_count,
            hidden,
            out_dim,
        ),
        WeightLayout::Bf16Packed => kernels::dispatch_linear_decode_cached_bf16_weights(
            vk_device,
            &dispatch_x,
            weight_buffer,
            row_count,
            hidden,
            out_dim,
        ),
    }
    .context("dispatch_forward_only: kernel dispatch")?;
    let mut out_dims = dims;
    *out_dims.last_mut().unwrap() = out_dim;
    out.reshape(out_dims.as_slice())
        .context("dispatch_forward_only: reshape out")
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_vulkan_kernel::{
        VulkanDevice,
        kernels::{upload_tensor_bf16_packed_buffer, upload_tensor_f32_buffer},
    };

    /// Synthetic parity test: a small `[T, H] @ [H, D]` matmul on
    /// Vulkan must match candle CPU's `broadcast_matmul` to the
    /// f32 numerics tolerance documented in
    /// `kiln-flce-kernel/src/tests.rs`.
    ///
    /// Skipped if no Vulkan device is available (CI without GPU).
    #[test]
    fn vulkan_linear_forward_parity_small() -> Result<()> {
        let Ok(vk_device) = VulkanDevice::new() else {
            eprintln!(
                "vulkan_linear_forward_parity_small: no Vulkan device available, skipping"
            );
            return Ok(());
        };
        let vk_device = Arc::new(vk_device);

        let device = Device::Cpu;
        let t = 5usize;
        let hidden = 8usize;
        let out_dim = 6usize;

        // Deterministic small inputs.
        let x_data: Vec<f32> = (0..t * hidden).map(|i| (i as f32) * 0.01).collect();
        let w_data: Vec<f32> = (0..hidden * out_dim).map(|i| (i as f32) * 0.02).collect();

        let x = Tensor::from_vec(x_data, (1, t, hidden), &device)?;
        let weight_t = Tensor::from_vec(w_data, (hidden, out_dim), &device)?;

        // CPU baseline.
        let baseline = x.broadcast_matmul(&weight_t)?;

        // Vulkan path.
        let weight_buffer = Arc::new(upload_tensor_f32_buffer(vk_device.as_ref(), &weight_t)?);
        let vulkan_out = dispatch_forward_only(
            vk_device.as_ref(),
            &x,
            weight_buffer.as_ref(),
            WeightLayout::F32,
            hidden,
            out_dim,
        )?;

        assert_eq!(vulkan_out.dims(), baseline.dims());
        let baseline_v = baseline.flatten_all()?.to_vec1::<f32>()?;
        let vulkan_v = vulkan_out.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(baseline_v.len(), vulkan_v.len());
        for (i, (b, v)) in baseline_v.iter().zip(vulkan_v.iter()).enumerate() {
            let abs = (b - v).abs();
            let rel = abs / (b.abs().max(1e-3));
            assert!(
                abs < 1e-3 || rel < 1e-3,
                "mismatch at idx {i}: baseline={b:.6} vulkan={v:.6} abs_diff={abs:e}"
            );
        }
        Ok(())
    }

    /// Same parity check but with the bf16-packed weight layout: the
    /// weight is uploaded via the bf16 packing path and the kernel
    /// expands each lane on the fly, matching the production
    /// inference path. Tighter tolerance (5e-3) reflects bf16's
    /// 7-bit mantissa.
    #[test]
    fn vulkan_linear_forward_parity_bf16_packed() -> Result<()> {
        let Ok(vk_device) = VulkanDevice::new() else {
            eprintln!("no Vulkan device, skipping");
            return Ok(());
        };
        let vk_device = Arc::new(vk_device);

        let device = Device::Cpu;
        let t = 4usize;
        let hidden = 8usize;
        let out_dim = 6usize;

        let x_data: Vec<f32> = (0..t * hidden).map(|i| (i as f32) * 0.01).collect();
        let w_data: Vec<f32> = (0..hidden * out_dim).map(|i| (i as f32) * 0.02).collect();

        let x = Tensor::from_vec(x_data, (1, t, hidden), &device)?;
        let weight_t_f32 = Tensor::from_vec(w_data, (hidden, out_dim), &device)?;
        let weight_t_bf16 = weight_t_f32.to_dtype(DType::BF16)?;

        // Baseline mirrors lora_loader::linear_with_lora_t's CPU
        // promote-to-f32 path so the comparison is apples-to-apples.
        let baseline = x.to_dtype(DType::F32)?.broadcast_matmul(
            &weight_t_bf16.to_dtype(DType::F32)?,
        )?;

        let weight_buffer = Arc::new(upload_tensor_bf16_packed_buffer(
            vk_device.as_ref(),
            &weight_t_bf16,
        )?);
        let vulkan_out = dispatch_forward_only(
            vk_device.as_ref(),
            &x,
            weight_buffer.as_ref(),
            WeightLayout::Bf16Packed,
            hidden,
            out_dim,
        )?;

        assert_eq!(vulkan_out.dims(), baseline.dims());
        let baseline_v = baseline.flatten_all()?.to_vec1::<f32>()?;
        let vulkan_v = vulkan_out.flatten_all()?.to_vec1::<f32>()?;
        for (i, (b, v)) in baseline_v.iter().zip(vulkan_v.iter()).enumerate() {
            let abs = (b - v).abs();
            let rel = abs / (b.abs().max(1e-3));
            assert!(
                abs < 5e-3 || rel < 5e-3,
                "bf16 mismatch at idx {i}: baseline={b:.6} vulkan={v:.6} abs_diff={abs:e}"
            );
        }
        Ok(())
    }

    /// End-to-end autograd parity: applying VulkanLinearOp via
    /// `apply_op1` and then calling `.backward()` must produce the
    /// same gradient on `x` as candle's native broadcast_matmul.
    /// This is the contract that lets the wrapper be safely wired
    /// into the training forward path — without it, training would
    /// silently produce wrong LoRA updates.
    #[test]
    fn vulkan_linear_backward_parity_small() -> Result<()> {
        let Ok(vk_device) = VulkanDevice::new() else {
            eprintln!("no Vulkan device, skipping");
            return Ok(());
        };
        let vk_device = Arc::new(vk_device);

        let device = Device::Cpu;
        let t = 4usize;
        let hidden = 6usize;
        let out_dim = 5usize;

        let x_data: Vec<f32> = (0..t * hidden).map(|i| 0.05 * (i as f32 + 1.0)).collect();
        let w_data: Vec<f32> = (0..hidden * out_dim)
            .map(|i| 0.03 * (i as f32 + 1.0))
            .collect();

        // The autograd-baseline path must mark x as requiring grad so
        // candle records it in the backprop graph; we use Var.
        let x_var = candle_core::Var::from_tensor(&Tensor::from_vec(
            x_data.clone(),
            (1, t, hidden),
            &device,
        )?)?;
        let weight_t = Tensor::from_vec(w_data.clone(), (hidden, out_dim), &device)?;

        // Baseline: candle native broadcast_matmul → loss = sum(out).
        let baseline_out = x_var.as_tensor().broadcast_matmul(&weight_t)?;
        let baseline_loss = baseline_out.sum_all()?;
        let baseline_grads = baseline_loss.backward()?;
        let baseline_dx = baseline_grads
            .get(x_var.as_tensor())
            .expect("baseline dx present")
            .clone();

        // Vulkan path: same x (fresh Var so the new graph stands alone).
        let x_var2 = candle_core::Var::from_tensor(&Tensor::from_vec(
            x_data, (1, t, hidden), &device,
        )?)?;
        let weight_buffer = Arc::new(upload_tensor_f32_buffer(vk_device.as_ref(), &weight_t)?);
        let op = build_op(
            vk_device.clone(),
            weight_buffer,
            weight_t.clone(),
            WeightLayout::F32,
            hidden,
            out_dim,
            DType::F32,
        );
        let vulkan_out = x_var2.as_tensor().apply_op1(op)?;
        let vulkan_loss = vulkan_out.sum_all()?;
        let vulkan_grads = vulkan_loss.backward()?;
        let vulkan_dx = vulkan_grads
            .get(x_var2.as_tensor())
            .expect("vulkan dx present")
            .clone();

        // Forward outputs must agree.
        let baseline_out_v = baseline_out.flatten_all()?.to_vec1::<f32>()?;
        let vulkan_out_v = vulkan_out.flatten_all()?.to_vec1::<f32>()?;
        for (i, (b, v)) in baseline_out_v.iter().zip(vulkan_out_v.iter()).enumerate() {
            let abs = (b - v).abs();
            assert!(
                abs < 1e-3,
                "fwd mismatch idx {i}: baseline={b:.6} vulkan={v:.6} abs_diff={abs:e}"
            );
        }

        // Gradients must agree.
        assert_eq!(baseline_dx.dims(), vulkan_dx.dims());
        let baseline_dx_v = baseline_dx.flatten_all()?.to_vec1::<f32>()?;
        let vulkan_dx_v = vulkan_dx.flatten_all()?.to_vec1::<f32>()?;
        for (i, (b, v)) in baseline_dx_v.iter().zip(vulkan_dx_v.iter()).enumerate() {
            let abs = (b - v).abs();
            let rel = abs / (b.abs().max(1e-3));
            assert!(
                abs < 1e-3 || rel < 1e-3,
                "bwd mismatch idx {i}: baseline={b:.6} vulkan={v:.6} abs_diff={abs:e}"
            );
        }
        Ok(())
    }

    /// Verify the buffer-offset kernel matches a contiguous chunk of the
    /// same weight tensor. Two dispatches against the same uploaded
    /// buffer (one with offset 0 over the full out_dim, one with a
    /// non-zero offset over a slice) should each match the candle
    /// reference for their respective slices.
    #[test]
    fn vulkan_linear_offset_parity() -> Result<()> {
        use kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_offset;

        let Ok(vk_device) = VulkanDevice::new() else {
            eprintln!("no Vulkan device, skipping");
            return Ok(());
        };

        let device = Device::Cpu;
        let t = 4usize;
        let hidden = 8usize;
        let full_out_dim = 12usize;
        let chunk_offset = 4usize;
        let chunk_len = 6usize;

        let x_data: Vec<f32> = (0..t * hidden).map(|i| (i as f32) * 0.01).collect();
        let w_data: Vec<f32> = (0..hidden * full_out_dim)
            .map(|i| (i as f32) * 0.02)
            .collect();
        let x = Tensor::from_vec(x_data, (1, t, hidden), &device)?;
        let weight_full = Tensor::from_vec(w_data, (hidden, full_out_dim), &device)?;
        let weight_full_bf16 = weight_full.to_dtype(DType::BF16)?;

        // Upload the full bf16-packed buffer once.
        let weight_buffer =
            upload_tensor_bf16_packed_buffer(&vk_device, &weight_full_bf16)?;

        // Chunk slice via the offset variant. The kernel returns
        // [batch_rows, 1, out_dim]; reshape to match the reference.
        let chunk_out_raw = dispatch_linear_decode_cached_bf16_weights_offset(
            &vk_device,
            &x,
            &weight_buffer,
            t,
            hidden,
            chunk_len,
            chunk_offset,
            full_out_dim,
        )?;
        let chunk_out = chunk_out_raw.reshape((1, t, chunk_len))?;

        // Reference: do the matmul against the same slice on CPU.
        let weight_chunk_bf16 =
            weight_full_bf16.narrow(1, chunk_offset, chunk_len)?.contiguous()?;
        let baseline_chunk = x
            .to_dtype(DType::F32)?
            .broadcast_matmul(&weight_chunk_bf16.to_dtype(DType::F32)?)?;

        assert_eq!(chunk_out.dims(), baseline_chunk.dims());
        let baseline_v = baseline_chunk.flatten_all()?.to_vec1::<f32>()?;
        let vulkan_v = chunk_out.flatten_all()?.to_vec1::<f32>()?;
        for (i, (b, v)) in baseline_v.iter().zip(vulkan_v.iter()).enumerate() {
            let abs = (b - v).abs();
            let rel = abs / (b.abs().max(1e-3));
            assert!(
                abs < 5e-3 || rel < 5e-3,
                "offset bf16 mismatch at idx {i}: baseline={b:.6} vulkan={v:.6} abs_diff={abs:e}"
            );
        }
        Ok(())
    }

    /// Verify the transposed-weight kernel matches `x @ W.T` against
    /// the same buffer the forward kernel uses. Used by
    /// VulkanLinearOp::bwd to compute dx without re-uploading W.T.
    #[test]
    fn vulkan_linear_transposed_parity() -> Result<()> {
        use kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_transposed;

        let Ok(vk_device) = VulkanDevice::new() else {
            eprintln!("no Vulkan device, skipping");
            return Ok(());
        };

        let device = Device::Cpu;
        let batch = 5usize;
        let forward_k = 8usize; // = bwd's n_dim
        let forward_n = 6usize; // = bwd's k_dim

        // Build W as a [forward_k, forward_n] row-major matrix (the
        // layout the forward kernel uses), upload as bf16-packed.
        let w_data: Vec<f32> = (0..forward_k * forward_n)
            .map(|i| (i as f32) * 0.03)
            .collect();
        let weight_full = Tensor::from_vec(w_data.clone(), (forward_k, forward_n), &device)?;
        let weight_full_bf16 = weight_full.to_dtype(DType::BF16)?;
        let weight_buffer = upload_tensor_bf16_packed_buffer(&vk_device, &weight_full_bf16)?;

        // x has shape [batch, k_dim] = [batch, forward_n].
        let x_data: Vec<f32> = (0..batch * forward_n)
            .map(|i| (i as f32) * 0.05)
            .collect();
        let x = Tensor::from_vec(x_data, (batch, forward_n), &device)?;

        // Vulkan: out = x @ W.T  (W.T shape [forward_n, forward_k] →
        // out shape [batch, forward_k]).
        let out_raw = dispatch_linear_decode_cached_bf16_weights_transposed(
            &vk_device,
            &x,
            &weight_buffer,
            batch,
            forward_n, // k_dim (inner sum)
            forward_k, // n_dim (output dim)
        )?;
        let out = out_raw.reshape((batch, forward_k))?;

        // Reference: candle CPU broadcast_matmul of x (f32) @ W.T (f32).
        let weight_t_f32 = weight_full_bf16
            .to_dtype(DType::F32)?
            .transpose(0, 1)?
            .contiguous()?;
        let baseline = x.broadcast_matmul(&weight_t_f32)?;

        assert_eq!(out.dims(), baseline.dims());
        let baseline_v = baseline.flatten_all()?.to_vec1::<f32>()?;
        let vulkan_v = out.flatten_all()?.to_vec1::<f32>()?;
        for (i, (b, v)) in baseline_v.iter().zip(vulkan_v.iter()).enumerate() {
            let abs = (b - v).abs();
            let rel = abs / (b.abs().max(1e-3));
            assert!(
                abs < 5e-3 || rel < 5e-3,
                "transposed bf16 mismatch at idx {i}: baseline={b:.6} vulkan={v:.6} abs_diff={abs:e}"
            );
        }
        Ok(())
    }

    /// Verify the Qwen3.5-style RMSNorm Vulkan kernel matches the
    /// candle CPU reference implementation in `kiln-model`. Uses the
    /// same `(1 + w) * x * rsqrt(mean(x^2) + eps)` semantics as
    /// `forward::rms_norm_fallback`.
    #[test]
    fn vulkan_qwen_rmsnorm_forward_parity() -> Result<()> {
        use kiln_vulkan_kernel::kernels::dispatch_qwen_rmsnorm_forward;

        let Ok(vk_device) = VulkanDevice::new() else {
            eprintln!("no Vulkan device, skipping");
            return Ok(());
        };

        let device = Device::Cpu;
        let rows = 5usize;
        let hidden = 16usize;
        let eps = 1e-6f32;

        let x_data: Vec<f32> = (0..rows * hidden)
            .map(|i| 0.05 * ((i as f32) + 1.0))
            .collect();
        let w_data: Vec<f32> = (0..hidden).map(|i| 0.01 * (i as f32)).collect();

        let x = Tensor::from_vec(x_data, (rows, hidden), &device)?;
        let weight = Tensor::from_vec(w_data, (hidden,), &device)?;

        // Vulkan path.
        let vulkan_out = dispatch_qwen_rmsnorm_forward(&vk_device, &x, &weight, eps)?;

        // CPU baseline mirroring rms_norm_fallback.
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rms_inv = (variance + eps as f64)?.sqrt()?.recip()?;
        let normed = x.broadcast_mul(&rms_inv)?;
        let one_plus_w = (weight.ones_like()? + &weight)?;
        let baseline = normed.broadcast_mul(&one_plus_w)?;

        assert_eq!(vulkan_out.dims(), baseline.dims());
        let baseline_v = baseline.flatten_all()?.to_vec1::<f32>()?;
        let vulkan_v = vulkan_out.flatten_all()?.to_vec1::<f32>()?;
        for (i, (b, v)) in baseline_v.iter().zip(vulkan_v.iter()).enumerate() {
            let abs = (b - v).abs();
            let rel = abs / (b.abs().max(1e-3));
            assert!(
                abs < 1e-4 || rel < 1e-4,
                "rmsnorm mismatch at idx {i}: baseline={b:.6} vulkan={v:.6} abs_diff={abs:e}"
            );
        }
        Ok(())
    }

    /// Op-state debug snapshot. Cheap sanity check that the struct
    /// formats useful metadata without panicking.
    #[test]
    fn vulkan_linear_op_debug_format() {
        let Ok(vk_device) = VulkanDevice::new() else {
            eprintln!("no Vulkan device, skipping");
            return;
        };
        let vk_device = Arc::new(vk_device);
        let device = Device::Cpu;
        let weight_t =
            Tensor::from_vec(vec![0.0f32; 4], (2, 2), &device).expect("build weight");
        let weight_buffer =
            Arc::new(upload_tensor_f32_buffer(vk_device.as_ref(), &weight_t).expect("upload"));
        let op = build_op(
            vk_device,
            weight_buffer,
            weight_t.clone(),
            WeightLayout::F32,
            2,
            2,
            DType::F32,
        );
        let s = format!("{op:?}");
        assert!(s.contains("VulkanLinearOp"));
        assert!(s.contains("hidden"));
    }
}
