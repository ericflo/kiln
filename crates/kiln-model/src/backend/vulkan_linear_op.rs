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
#[cfg(test)]
use candle_core::Device;
use candle_core::op::BackpropOp;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, Shape, Storage, Tensor};

use kiln_vulkan_kernel::{VulkanBuffer, VulkanDevice, kernels};

/// Per-dispatch FLOP ceiling for the Vulkan-routed matmul. Above this,
/// `cpu_fwd` and `bwd` chunk the dispatch into multiple submits (BF16
/// path) or fall back to candle's CPU `broadcast_matmul` (F32 path)
/// rather than queuing a single Vulkan submit large enough to risk
/// hanging the GPU/driver.
///
/// **Why:** the host hard-hung twice during the original
/// `/tmp/sft-data.jsonl` SFT repro (T≈918, vocab=152064) — kernel logs
/// went silent (no OOM, no AMDGPU reset, no panic), the box had to be
/// physically rebooted both times. The implicated dispatch was the
/// training-time lm_head forward `[918, 2560] @ [2560, 152064]` which
/// queues ~4.36M workgroups (32-col tiles × 918 batch rows) in one
/// submit on a 40-CU APU. Each workgroup also walks a 2560-element
/// inner reduction, so total bandwidth is enormous and AMD's recovery
/// path on Strix Halo evidently does not handle the queue depth
/// gracefully.
///
/// **Calibration:** FLCE has been running with chunk_size=4096 vocab
/// columns at T~244 without issue — that's `244 * 2560 * 4096 * 2`
/// = ~5 GFLOP per submit. At T=918 the same chunk shape is ~19 GFLOP.
/// Both are safe in observed practice. Setting the ceiling at 20
/// GFLOP gives ≈ 800 ms per submit at 25 TFLOPS — comparable to FLCE's
/// proven cadence, leaves frequent compositor preemption windows, and
/// produces a manageable chunk count (lm_head fwd at T=918 splits into
/// ~38 chunks, same as FLCE's vocab-chunk count, so total dispatch
/// overhead is bounded).
///
/// Tunable via `KILN_VULKAN_LINEAR_MAX_GFLOP` (parsed once at process
/// start). Set to 0 to disable the guard entirely (NOT recommended on
/// Strix Halo). Decimals are accepted (e.g. `KILN_VULKAN_LINEAR_MAX_GFLOP=50.0`).
const DEFAULT_MAX_FLOP_PER_DISPATCH: u64 = 20_000_000_000;

/// FLOP estimate for a `[batch, hidden] @ [hidden, out_dim]` matmul.
/// Counts one multiply + one add per inner term, matching the kernel's
/// per-iteration work.
fn matmul_flop(batch: usize, hidden: usize, out_dim: usize) -> u64 {
    (batch as u64)
        .saturating_mul(hidden as u64)
        .saturating_mul(out_dim as u64)
        .saturating_mul(2)
}

fn max_flop_per_dispatch() -> u64 {
    static CEILING: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CEILING.get_or_init(|| {
        std::env::var("KILN_VULKAN_LINEAR_MAX_GFLOP")
            .ok()
            .as_deref()
            .map(str::trim)
            .and_then(|s| s.parse::<f64>().ok())
            .map(|gflop| {
                if gflop <= 0.0 {
                    u64::MAX
                } else {
                    (gflop * 1.0e9_f64).round() as u64
                }
            })
            .unwrap_or(DEFAULT_MAX_FLOP_PER_DISPATCH)
    })
}

/// True when the requested matmul shape would exceed the per-dispatch
/// FLOP ceiling. For the BF16-packed weight path the caller can split
/// the matmul into per-chunk submits via [`max_chunk_dim_for_flop`];
/// for the F32 weight path there is no offset kernel and the caller
/// should fall back to CPU `broadcast_matmul` (the pre-Phase-2 baseline
/// that did not crash).
pub fn dispatch_exceeds_safety_ceiling(batch: usize, hidden: usize, out_dim: usize) -> bool {
    matmul_flop(batch, hidden, out_dim) > max_flop_per_dispatch()
}

/// Largest `chunk_dim` such that
/// `2 × other_dim_product × chunk_dim ≤ max_flop_per_dispatch()`.
///
/// Used by [`VulkanLinearOp::cpu_fwd`] to chunk oversized forward
/// matmuls along the output dim (`other_dim_product = batch × hidden`)
/// and by [`VulkanLinearOp::bwd`] to chunk oversized backward matmuls
/// along the batch dim (`other_dim_product = out_dim × hidden`).
///
/// Always returns at least 1 — even a microscopic ceiling shouldn't
/// produce zero-sized chunks (the caller's loop would never advance).
/// Saturates: if `other_dim_product` is so large that even one element
/// of the chunked dim would exceed the ceiling, returns 1 (caller does
/// one element at a time, which is slow but safe).
pub fn max_chunk_dim_for_flop(other_dim_product: usize) -> usize {
    let max_flop = max_flop_per_dispatch();
    if max_flop == u64::MAX {
        // Guard disabled — caller will dispatch single-shot.
        return usize::MAX;
    }
    let denom = (other_dim_product as u64).saturating_mul(2).max(1);
    let chunk = (max_flop / denom) as usize;
    chunk.max(1)
}

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

    fn cpu_fwd(&self, s_x: &CpuStorage, l_x: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
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
                .map_err(|e| candle_core::Error::Msg(format!("VulkanLinearOp reshape x: {e:?}")))?
        };

        // Decide single-shot vs chunked dispatch. The BF16-packed path
        // has an offset kernel, so oversized matmuls split along the
        // out_dim with bounded per-submit work — each chunk dispatch
        // calls `queue_wait_idle()` on completion, giving the display
        // compositor preemption points between chunks. The F32 path has
        // no offset kernel; its caller (`linear_prefill_apply`) bails
        // to CPU `broadcast_matmul` before we get here, so a single-
        // shot F32 dispatch reaching this point is by construction
        // already under the safety ceiling.
        let oversized = dispatch_exceeds_safety_ceiling(row_count, self.hidden, self.out_dim);
        let out_tensor = if oversized && self.weight_layout == WeightLayout::Bf16Packed {
            let other_dims = row_count.saturating_mul(self.hidden);
            let chunk_out_dim = max_chunk_dim_for_flop(other_dims);
            let chunk_count = self.out_dim.div_ceil(chunk_out_dim);
            // One trace per process for the chunked path so the operator
            // can confirm chunking is engaging without per-step log spam.
            // The first chunked dispatch is the most informative one
            // (covers lm_head at training T) — subsequent dispatches
            // re-check the OnceLock and short-circuit.
            static FIRST_CHUNKED_LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            FIRST_CHUNKED_LOGGED.get_or_init(|| {
                tracing::info!(
                    row_count,
                    hidden = self.hidden,
                    out_dim = self.out_dim,
                    total_gflop = matmul_flop(row_count, self.hidden, self.out_dim) as f64 / 1.0e9,
                    chunk_out_dim,
                    chunk_count,
                    per_chunk_gflop =
                        matmul_flop(row_count, self.hidden, chunk_out_dim) as f64 / 1.0e9,
                    "VulkanLinearOp::cpu_fwd first chunked dispatch"
                );
            });
            let mut chunk_outputs: Vec<Tensor> = Vec::new();
            let mut chunk_start = 0usize;
            while chunk_start < self.out_dim {
                let chunk_len = (self.out_dim - chunk_start).min(chunk_out_dim);
                let chunk_out = kernels::dispatch_linear_decode_cached_bf16_weights_offset(
                    self.vk_device.as_ref(),
                    &dispatch_x,
                    self.weight_buffer.as_ref(),
                    row_count,
                    self.hidden,
                    chunk_len,
                    chunk_start,
                    self.out_dim,
                )
                .map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "VulkanLinearOp chunked dispatch (start={chunk_start}, \
                         len={chunk_len}, full={}): {e:?}",
                        self.out_dim
                    ))
                })?;
                chunk_outputs.push(chunk_out);
                chunk_start += chunk_len;
            }
            // Each chunk output is `[row_count, 1, chunk_len]`. Concat
            // along the last dim reproduces the single-shot output.
            Tensor::cat(&chunk_outputs, 2).map_err(|e| {
                candle_core::Error::Msg(format!(
                    "VulkanLinearOp chunked concat ({} chunks): {e:?}",
                    chunk_outputs.len()
                ))
            })?
        } else {
            match self.weight_layout {
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
            .map_err(|e| candle_core::Error::Msg(format!("VulkanLinearOp dispatch: {e:?}")))?
        };

        // Restore the original leading dims with `out_dim` swapped in for
        // `inner`. CustomOp1's output Shape is what candle uses to size
        // downstream ops, so it has to match what a `broadcast_matmul`
        // would have produced.
        let mut out_dims: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_dim);
        let out_tensor = out_tensor
            .reshape(out_dims.as_slice())
            .map_err(|e| candle_core::Error::Msg(format!("VulkanLinearOp reshape out: {e:?}")))?;

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
        // re-upload of a transposed view). Oversized dispatches are
        // chunked along the batch dim — the transposed kernel itself
        // takes a `batch` parameter, so we just call it N times with
        // disjoint slices of `grad_y` and concat along axis 0.
        // Each per-chunk dispatch calls `queue_wait_idle()` on
        // completion (compositor preemption), keeping the GPU
        // submit-time bounded. F32 weights have no transposed kernel
        // and fall through to the CPU `broadcast_matmul` path below.
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
                .map_err(|e| candle_core::Error::Msg(format!("bwd reshape grad_y: {e:?}")))?;
            let oversized = dispatch_exceeds_safety_ceiling(row_count, self.out_dim, self.hidden);
            let dx_3d = if oversized {
                let other_dims = self.out_dim.saturating_mul(self.hidden);
                let chunk_batch = max_chunk_dim_for_flop(other_dims);
                let chunk_count = row_count.div_ceil(chunk_batch);
                static FIRST_CHUNKED_BWD_LOGGED: std::sync::OnceLock<()> =
                    std::sync::OnceLock::new();
                FIRST_CHUNKED_BWD_LOGGED.get_or_init(|| {
                    tracing::info!(
                        row_count,
                        out_dim = self.out_dim,
                        hidden = self.hidden,
                        total_gflop =
                            matmul_flop(row_count, self.out_dim, self.hidden) as f64 / 1.0e9,
                        chunk_batch,
                        chunk_count,
                        per_chunk_gflop =
                            matmul_flop(chunk_batch, self.out_dim, self.hidden) as f64 / 1.0e9,
                        "VulkanLinearOp::bwd first chunked dispatch"
                    );
                });
                let mut chunk_outputs: Vec<Tensor> = Vec::new();
                let mut chunk_start = 0usize;
                while chunk_start < row_count {
                    let chunk_len = (row_count - chunk_start).min(chunk_batch);
                    let chunk_dy = dispatch_x.narrow(0, chunk_start, chunk_len).map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "bwd narrow grad_y (start={chunk_start}, \
                                 len={chunk_len}): {e:?}"
                        ))
                    })?;
                    let chunk_dx = kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_transposed(
                        self.vk_device.as_ref(),
                        &chunk_dy,
                        self.weight_buffer.as_ref(),
                        chunk_len,
                        self.out_dim,
                        self.hidden,
                    )
                    .map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "bwd chunked transposed dispatch (start={chunk_start}, \
                             len={chunk_len}): {e:?}"
                        ))
                    })?;
                    chunk_outputs.push(chunk_dx);
                    chunk_start += chunk_len;
                }
                Tensor::cat(&chunk_outputs, 0).map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "bwd chunked concat ({} chunks): {e:?}",
                        chunk_outputs.len()
                    ))
                })?
            } else {
                kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_transposed(
                    self.vk_device.as_ref(),
                    &dispatch_x,
                    self.weight_buffer.as_ref(),
                    row_count,
                    self.out_dim,
                    self.hidden,
                )
                .map_err(|e| candle_core::Error::Msg(format!("bwd transposed dispatch: {e:?}")))?
            };
            // dx_3d is [row_count, 1, hidden]; restore caller's leading dims.
            let mut out_dims: Vec<usize> = dims[..dims.len() - 1].to_vec();
            out_dims.push(self.hidden);
            let dx_f32 = dx_3d
                .reshape(out_dims.as_slice())
                .map_err(|e| candle_core::Error::Msg(format!("bwd reshape dx: {e:?}")))?;
            let dx = if self.out_dtype == DType::F32 {
                dx_f32
            } else {
                dx_f32
                    .to_dtype(self.out_dtype)
                    .map_err(|e| candle_core::Error::Msg(format!("bwd cast dx: {e:?}")))?
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
            eprintln!("vulkan_linear_forward_parity_small: no Vulkan device available, skipping");
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
        let baseline = x
            .to_dtype(DType::F32)?
            .broadcast_matmul(&weight_t_bf16.to_dtype(DType::F32)?)?;

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
        let x_var2 =
            candle_core::Var::from_tensor(&Tensor::from_vec(x_data, (1, t, hidden), &device)?)?;
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
        let weight_buffer = upload_tensor_bf16_packed_buffer(&vk_device, &weight_full_bf16)?;

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
        let weight_chunk_bf16 = weight_full_bf16
            .narrow(1, chunk_offset, chunk_len)?
            .contiguous()?;
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
        let x_data: Vec<f32> = (0..batch * forward_n).map(|i| (i as f32) * 0.05).collect();
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

    /// Verify the RMSNorm backward Vulkan kernel matches candle's
    /// autograd gradient for the same forward.
    #[test]
    fn vulkan_qwen_rmsnorm_backward_parity() -> Result<()> {
        use kiln_vulkan_kernel::kernels::dispatch_qwen_rmsnorm_backward;

        let Ok(vk_device) = VulkanDevice::new() else {
            eprintln!("no Vulkan device, skipping");
            return Ok(());
        };

        let device = Device::Cpu;
        let rows = 4usize;
        let hidden = 12usize;
        let eps = 1e-6f32;

        let x_data: Vec<f32> = (0..rows * hidden)
            .map(|i| 0.05 * (((i % 7) + 1) as f32))
            .collect();
        let w_data: Vec<f32> = (0..hidden).map(|i| 0.01 * (i as f32 + 1.0)).collect();
        let grad_y_data: Vec<f32> = (0..rows * hidden)
            .map(|i| 0.02 * (((i % 5) as i32 - 2) as f32))
            .collect();

        let x = Tensor::from_vec(x_data.clone(), (rows, hidden), &device)?;
        let weight = Tensor::from_vec(w_data, (hidden,), &device)?;
        let grad_y = Tensor::from_vec(grad_y_data, (rows, hidden), &device)?;

        // Vulkan path.
        let vulkan_grad_x = dispatch_qwen_rmsnorm_backward(&vk_device, &x, &weight, &grad_y, eps)?;

        // Candle autograd reference: build forward as a candle graph
        // over a Var, compute loss = sum(y * grad_y), backward, read
        // dL/dx — which equals the requested gradient.
        let x_var = candle_core::Var::from_tensor(&x)?;
        let variance = x_var
            .as_tensor()
            .sqr()?
            .mean_keepdim(candle_core::D::Minus1)?;
        let rms_inv = (variance + eps as f64)?.sqrt()?.recip()?;
        let normed = x_var.as_tensor().broadcast_mul(&rms_inv)?;
        let one_plus_w = (weight.ones_like()? + &weight)?;
        let y = normed.broadcast_mul(&one_plus_w)?;
        // Synthetic loss = sum(y * grad_y) — its gradient w.r.t. y is grad_y.
        let synthetic_loss = (y * &grad_y)?.sum_all()?;
        let grads = synthetic_loss.backward()?;
        let baseline_grad_x = grads.get(x_var.as_tensor()).expect("dx present").clone();

        assert_eq!(vulkan_grad_x.dims(), baseline_grad_x.dims());
        let baseline_v = baseline_grad_x.flatten_all()?.to_vec1::<f32>()?;
        let vulkan_v = vulkan_grad_x.flatten_all()?.to_vec1::<f32>()?;
        for (i, (b, v)) in baseline_v.iter().zip(vulkan_v.iter()).enumerate() {
            let abs = (b - v).abs();
            let rel = abs / (b.abs().max(1e-3));
            assert!(
                abs < 1e-4 || rel < 1e-3,
                "rmsnorm bwd mismatch at idx {i}: baseline={b:.6} vulkan={v:.6} abs_diff={abs:e}"
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
        let weight_t = Tensor::from_vec(vec![0.0f32; 4], (2, 2), &device).expect("build weight");
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

    /// Regression for the second host hard-hang on the original
    /// `/tmp/sft-data.jsonl` repro: at T=918 the lm_head training-time
    /// forward queues `[918, 2560] @ [2560, 152064]` = ~715 GFLOP in a
    /// single Vulkan submit, which the Strix Halo APU could not handle
    /// — the kernel logs went silent and the box had to be physically
    /// rebooted. The safety ceiling MUST classify that shape as
    /// "do not dispatch single-shot"; the BF16-packed path uses this
    /// signal to switch to chunked dispatch (see `cpu_fwd` and `bwd`),
    /// the F32 path uses it to bail to CPU `broadcast_matmul`.
    #[test]
    fn safety_guard_rejects_lm_head_repro_shape() {
        // T=918 lm_head forward shape from the crashed repro.
        // ~715 GFLOP — vastly above the 20 GFLOP per-submit ceiling.
        assert!(
            dispatch_exceeds_safety_ceiling(918, 2560, 152064),
            "[918, 2560] @ [2560, 152064] is the shape that hung the host \
             twice; the BF16 path must classify it as 'chunk this'"
        );
        // Even modest lm_head shapes get chunked — at T=16 the matmul
        // is 12 GFLOP (under), at T=32 it's 25 GFLOP (over). The
        // chunking is essentially free above ~16 since the per-submit
        // overhead is microseconds vs hundreds of milliseconds of
        // compute per chunk.
        assert!(!dispatch_exceeds_safety_ceiling(16, 2560, 152064));
        assert!(dispatch_exceeds_safety_ceiling(32, 2560, 152064));
    }

    /// Counter-test: small projection shapes should still single-shot
    /// dispatch — they're well under the per-submit ceiling. Larger
    /// projections (GDN in_proj at T=918, MLP gate_up) split into a
    /// handful of chunks, which is fine: per-dispatch overhead is
    /// ~50µs each vs the ~800ms compute per chunk.
    #[test]
    fn safety_guard_allows_normal_projection_shapes() {
        // Qwen3.5-4B q_proj forward: T × hidden=2560 → out_dim=2560.
        // 12 GFLOP at T=918, well under the 20 GFLOP ceiling.
        for t in [1, 64, 256, 918] {
            assert!(
                !dispatch_exceeds_safety_ceiling(t, 2560, 2560),
                "q_proj-shape [T={t}, 2560, 2560] is well under the \
                 ceiling — should single-shot dispatch"
            );
        }
        // At T=2048 q_proj would be 27 GFLOP (over), so chunking
        // engages — that's expected and safe.
        assert!(dispatch_exceeds_safety_ceiling(2048, 2560, 2560));

        // GDN in_proj_qkv at T=918 (~38 GFLOP) and MLP gate_up
        // (~24 GFLOP) chunk into a handful of submits. They are
        // OVER the per-submit ceiling but UNDER the multi-chunk
        // overhead concern (a few chunks each).
        assert!(dispatch_exceeds_safety_ceiling(918, 2560, 8192));
        assert!(dispatch_exceeds_safety_ceiling(918, 2560, 10240));
    }

    /// Env-var override: `KILN_VULKAN_LINEAR_MAX_GFLOP=0` should make
    /// the guard a no-op (every shape allowed). Tested via the
    /// `matmul_flop` arithmetic — the env-parsing path is covered
    /// by the OnceLock initializer, but we don't invoke it from a
    /// test (would race with other tests that observe the default).
    #[test]
    fn matmul_flop_arithmetic_is_correct() {
        // Standard case: 2 * batch * hidden * out_dim FMAs.
        assert_eq!(matmul_flop(2, 3, 4), 2 * 2 * 3 * 4);
        assert_eq!(matmul_flop(918, 2560, 152064), 2u64 * 918 * 2560 * 152064);
        // Saturating cases for paranoia: large multiply must not
        // panic or overflow silently in debug builds.
        let big = matmul_flop(usize::MAX, 1, 1);
        assert_eq!(big, u64::MAX, "must saturate, not overflow");
    }

    /// Chunk-sizing arithmetic: the chunk dim returned must keep each
    /// per-chunk dispatch under the FLOP ceiling, and must always be
    /// at least 1 (a zero-length chunk would cause an infinite loop
    /// in the caller's chunking loop).
    #[test]
    fn max_chunk_dim_for_flop_is_within_ceiling() {
        // Default 20 GFLOP per-submit ceiling. lm_head fwd chunking
        // shape: other_dims = batch * hidden = 918 * 2560.
        let chunk = max_chunk_dim_for_flop(918 * 2560);
        let per_chunk_flop = matmul_flop(918, 2560, chunk);
        assert!(
            per_chunk_flop <= max_flop_per_dispatch(),
            "lm_head fwd chunk_size={chunk} → {per_chunk_flop} FLOP > ceiling \
             {}",
            max_flop_per_dispatch()
        );
        // Sanity: at the 20 GFLOP ceiling, chunk size for lm_head
        // fwd should be approximately the FLCE chunk size (4096 vocab
        // cols), since the underlying compute and target per-submit
        // time are the same. Loose bound — must be in [1024, 8192].
        assert!(
            (1024..=8192).contains(&chunk),
            "lm_head fwd chunk size {chunk} should be in FLCE-ish range"
        );

        // For the lm_head bwd shape: other_dims = out_dim * hidden
        // = 152064 * 2560 (huge), chunk_batch should be small but ≥ 1.
        let chunk_b = max_chunk_dim_for_flop(152064 * 2560);
        let per_chunk_flop_b = matmul_flop(chunk_b, 152064, 2560);
        assert!(per_chunk_flop_b <= max_flop_per_dispatch());
        assert!(chunk_b >= 1);
    }

    /// Synthetic parity test for the in-op chunking path. The
    /// chunked-dispatch output for an oversized shape must match the
    /// single-shot output of the same op evaluated at a smaller (but
    /// otherwise identical) shape — proven by running the SAME op
    /// twice with different `KILN_VULKAN_LINEAR_MAX_GFLOP`. This test
    /// instead exercises a pure-arithmetic invariant: chunking should
    /// be a no-op when each chunk fits in one dispatch — which is
    /// the regression we'd want to catch if `Tensor::cat` behaved
    /// differently than the single-shot kernel output for the
    /// degenerate case of one chunk == full output dim.
    #[test]
    fn vulkan_linear_chunked_fwd_parity_small() -> Result<()> {
        let Ok(vk_device) = VulkanDevice::new() else {
            eprintln!(
                "vulkan_linear_chunked_fwd_parity_small: no Vulkan device available, skipping"
            );
            return Ok(());
        };
        let vk_device = Arc::new(vk_device);
        let device = Device::Cpu;
        let t = 4usize;
        let hidden = 8usize;
        let out_dim = 12usize;
        // Use the bf16-packed path (the only one with chunking).
        let x_data: Vec<f32> = (0..t * hidden).map(|i| (i as f32) * 0.013).collect();
        let w_data: Vec<f32> = (0..hidden * out_dim).map(|i| (i as f32) * 0.017).collect();
        let x = Tensor::from_vec(x_data, (1, t, hidden), &device)?;
        let weight_t_f32 = Tensor::from_vec(w_data, (hidden, out_dim), &device)?;
        let weight_t_bf16 = weight_t_f32.to_dtype(DType::BF16)?;
        let baseline = x.broadcast_matmul(&weight_t_bf16.to_dtype(DType::F32)?)?;

        let weight_buffer = Arc::new(upload_tensor_bf16_packed_buffer(
            vk_device.as_ref(),
            &weight_t_bf16,
        )?);

        // Single-shot Vulkan dispatch.
        let single_shot = dispatch_forward_only(
            vk_device.as_ref(),
            &x,
            weight_buffer.as_ref(),
            WeightLayout::Bf16Packed,
            hidden,
            out_dim,
        )?;

        // Manually-chunked Vulkan dispatch (3 chunks of 4 cols each).
        // This mimics what `cpu_fwd` does when it crosses the FLOP
        // ceiling. We don't go through the env-var path because
        // `OnceLock` would lock in whatever the first observation is
        // and pollute neighboring tests.
        let dispatch_x = if x.dtype() == DType::F32 {
            x.clone()
        } else {
            x.to_dtype(DType::F32)?
        };
        let mut chunks: Vec<Tensor> = Vec::new();
        let chunk_size = 4usize;
        let mut start = 0usize;
        while start < out_dim {
            let len = (out_dim - start).min(chunk_size);
            let chunk =
                kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_offset(
                    vk_device.as_ref(),
                    &dispatch_x,
                    weight_buffer.as_ref(),
                    t,
                    hidden,
                    len,
                    start,
                    out_dim,
                )?;
            chunks.push(chunk);
            start += len;
        }
        let chunked = Tensor::cat(&chunks, 2)?;

        // All three should agree to f32 precision.
        let baseline_v = baseline.flatten_all()?.to_vec1::<f32>()?;
        let single_v = single_shot.flatten_all()?.to_vec1::<f32>()?;
        let chunked_v = chunked.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(baseline_v.len(), single_v.len());
        assert_eq!(baseline_v.len(), chunked_v.len());
        for (i, ((b, s), c)) in baseline_v
            .iter()
            .zip(single_v.iter())
            .zip(chunked_v.iter())
            .enumerate()
        {
            // Single-shot vs chunked: must be bit-exact (same kernel
            // family, same accumulator order).
            assert!(
                (s - c).abs() < 1e-6,
                "single_shot vs chunked diverge at idx {i}: \
                 single={s:.6} chunked={c:.6}"
            );
            // Baseline f32 vs Vulkan bf16-weight: small drift OK.
            assert!(
                (b - c).abs() < 1e-2,
                "baseline vs chunked drift at idx {i}: baseline={b:.6} chunked={c:.6}"
            );
        }
        Ok(())
    }
}
