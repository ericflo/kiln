//! PR4 — `VkBwdAdapter`: bridge a Vulkan leaf-kernel backward
//! (`kiln_vulkan_kernel::vk_tensor::VkBackwardOp`) into the shared autograd
//! tape (`kiln_autograd::BackwardOp` / `kiln_autograd::Tape`).
//!
//! The Vulkan leaf kernels already carry a complete eager autograd: every
//! forward op family records an `Arc<dyn VkBackwardOp>` that owns Arc-cloned
//! input `VkTensor`s and computes input grads via on-device kernels
//! (`MatmulBackward`, `RmsNormBackward`, `RopeBackward`,
//! `SoftmaxLastDimBackward`, …). The shared substrate walks a *different*
//! graph: a `kiln_autograd::Tape` of `Box<dyn BackwardOp>` over
//! `kiln_tensor::Tensor`. This adapter presents a `VkBackwardOp` as a
//! `BackwardOp`: on `apply`, it zero-copy-bridges the incoming `grad_output`
//! `Tensor` to a `VkTensor` (`kiln_tensor::vk_tensor_from_kt` — an
//! `Arc<VulkanBuffer>` refcount bump, no D2H/H2D), runs the existing vk_ops
//! backward kernel, and rewraps each returned `VkTensor` back into a
//! `Tensor(VulkanStorage)` (`kiln_tensor::kt_tensor_from_vk`), preserving the
//! forward input order and the `None` slots exactly.
//!
//! This mirrors `kiln-opd-loss-kernel/src/kt_tape.rs` (the CUDA/Metal OPD
//! reverse-KL composite), which bridges a fused kernel backward into the same
//! shared tape: saved tensors + host metadata live inside the wrapped backward
//! struct, and `apply` only translates the grad tensor at the boundary.
//!
//! The **one structural difference** from the OPD kt_tape: that one is a single
//! composite that re-derives `d_hidden` analytically; the Vulkan families are
//! per-op-family fused kernels. So this adapter is **GENERIC** — a single
//! `impl BackwardOp` reused across all families. Op-family granularity does not
//! live in the adapter; it lives in [`family_ported`] (the no-flag-day switch
//! the PR5 forward recorder consults) plus the PR5 recorder itself. Un-listed
//! families stay on the vk-native `vk_autograd::vk_backward` walk.
//!
//! Nothing in this module is wired yet: it adds a new, not-yet-recorded type.
//! The PR5 recorders will `tape.record(&out, &inputs, Box::new(VkBwdAdapter(gf)))`
//! for families [`family_ported`] returns `true` for.

use std::sync::Arc;

use kiln_autograd::BackwardOp;
use kiln_tensor::{Device, Error, Result, Tensor};
use kiln_vulkan_kernel::vk_tensor::VkBackwardOp;

/// Presents a leaf [`VkBackwardOp`] to the shared `kiln_autograd::Tape`.
///
/// Generic over `Arc<dyn VkBackwardOp>`: one impl bridges *all* op families.
/// The wrapped backward already owns its saved input `VkTensor`s (and, for
/// fused families, any host-side metadata) — see the module doc. The adapter's
/// only job is to translate the grad tensor at the storage boundary, zero-copy,
/// and to preserve the input-order / `None`-slot contract that the tape walker
/// relies on to bind each returned grad to the right input `TensorId`.
#[derive(Debug)]
pub struct VkBwdAdapter(pub Arc<dyn VkBackwardOp>);

impl BackwardOp for VkBwdAdapter {
    fn name(&self) -> &'static str {
        self.0.op_name()
    }

    fn input_count(&self) -> usize {
        self.0.input_refs().len()
    }

    /// Bridge `grad_output` to a `VkTensor`, run the leaf kernel backward, and
    /// rewrap each returned grad — preserving length, slot order, and `None`
    /// positions exactly.
    ///
    /// Does NOT mutate `grad_output` or any saved input (the `BackwardOp`
    /// contract): `vk_tensor_from_kt` only refcount-bumps the underlying
    /// `Arc<VulkanBuffer>`, and the leaf kernels write their outputs to fresh
    /// device buffers.
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // Zero-copy: refcount-bump grad_output's Arc<VulkanBuffer> into a
        // VkTensor. Rejects non-Vulkan storage, non-contiguous / offset
        // layouts, and unsupported dtypes (F32/BF16 only) loudly.
        let vk_grad = kiln_tensor::vk_tensor_from_kt(grad_output)?;

        // The Vulkan ordinal to stamp on the rewrapped grad Tensors. Grads
        // flow back on the same device the grad_output arrived on; derive it
        // from grad_output so this generic adapter needs no construction-time
        // plumbing (and matches the required single-field tuple shape).
        //
        // Match `Device::Vulkan(i)` SPECIFICALLY: `Device::index()` also returns
        // `Some` for Cuda/Metal, which would make a non-Vulkan grad slip past
        // here only to fail less clearly inside `vk_tensor_from_kt`. (In practice
        // `vk_tensor_from_kt` above already rejects non-Vulkan storage, but the
        // ordinal we stamp must come from a *Vulkan* device or it is meaningless.)
        let device_index = match grad_output.device() {
            Device::Vulkan(i) => i,
            other => {
                return Err(Error::Msg(format!(
                    "VkBwdAdapter[{}]: grad_output is not on a Vulkan device (got {other})",
                    self.0.op_name()
                )));
            }
        };

        let vk_grads = self
            .0
            .backward(&vk_grad)
            .map_err(|e| Error::Msg(format!("VkBwdAdapter[{}] backward: {e}", self.0.op_name())))?;

        // Preserve length + slot order + None positions EXACTLY. The tape
        // walker binds the i-th returned grad to node.input_ids[i]; a dropped
        // or reordered None would silently mis-attribute a gradient. Thread the
        // Result through the Option with `.map(...).transpose()`; never
        // `.flatten()` / `.filter_map()`.
        vk_grads
            .into_iter()
            .map(|opt| {
                opt.map(|v| kiln_tensor::kt_tensor_from_vk(&v, device_index))
                    .transpose()
            })
            .collect()
    }
}

/// Families whose [`VkBwdAdapter`] is validated and may record onto the shared
/// `kiln_autograd::Tape`. Everything else falls through to the vk-native
/// `kiln_vulkan_kernel::vk_autograd::vk_backward` walk — no flag day.
///
/// This is the data-only switch the PR5 forward recorder consults: for a ported
/// family the Vulkan forward records a `VkBwdAdapter` onto the kt tape; for an
/// un-ported family it keeps recording the vk-native graph. PR4 ships the
/// registry + the generic adapter together; PR5 wires the registry into the
/// recorders. Nothing calls this yet.
///
/// `op_name` is the `VkBackwardOp::op_name()` literal (re-verified against the
/// `fn op_name` returns: `"matmul"`, `"rms_norm"`, `"rope"`,
/// `"softmax_lastdim"`). Extended op-family-by-op-family as each family's FD
/// parity (PR4b) lands.
///
/// Every family listed here is validated by a *finite-difference* grad test in
/// `tests/vk_bwd_adapter_parity.rs` (FD oracle vs the adapter-produced analytic
/// grad), not merely an exact-vs-direct identity — an FD test is what catches a
/// wrong backward *kernel*. PR5 will record these families onto the kt tape, so
/// an entry here is a correctness assertion.
///
/// `matmul_bf16w` is DELIBERATELY ABSENT: BF16 matmul has no grad-validation
/// test yet. Its kt<->vk bridge also needs care — the logical `byte_len` the
/// bridge records on the VulkanStorage must equal the BF16 *logical* byte size
/// (`n_elements * 2`), NOT the pool-padded device allocation (spec R5) — so a
/// BF16 grad round-trip can readback the wrong range if mis-bridged. Re-add
/// `"matmul_bf16w"` to this list only once a BF16 FD test in the parity suite
/// goes green.
pub fn family_ported(op_name: &str) -> bool {
    matches!(
        op_name,
        // Wave 1 — the core training path. F32-only families, FD-validated.
        "matmul" | "rms_norm" | "rope" | "softmax_lastdim" // Pending BF16 grad-validation (see doc above): "matmul_bf16w"
                                                           // Wave 2 appended here as each family's FD test goes green:
                                                           // | "flce" | "grpo" | "gdn_chunkwise" | "opd_topk_kl"
    )
}
