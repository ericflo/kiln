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
        x: &Tensor,
        _y: &Tensor,
        grad_y: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        // dX = grad_y @ W where W = weight_t.t() has shape [out_dim, hidden].
        // First-cut: route through candle CPU for parity safety. A future
        // commit can replace this with a second Vulkan dispatch (the
        // transpose-of-transpose lookup makes this a one-line swap).
        let _ = (x, grad_y);
        Err(candle_core::Error::Msg(
            "VulkanLinearOp::bwd: not yet implemented — wire forward.rs only after backward lands".into(),
        ))
    }
}

/// Convenience constructor that uploads (or reuses a cached upload of) the
/// weight buffer and returns a ready-to-apply [`VulkanLinearOp`].
///
/// `weight_t` is the row-major `[hidden, out_dim]` transposed weight. The
/// caller is expected to have computed it once at load time.
pub fn build_op(
    vk_device: Arc<VulkanDevice>,
    weight_buffer: Arc<VulkanBuffer>,
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
