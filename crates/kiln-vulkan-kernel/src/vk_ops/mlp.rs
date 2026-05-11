//! SwiGLU MLP block for VkTensor (forward + backward via composition).
//!
//! Standard SwiGLU: out = (silu(x @ W_gate.T) * (x @ W_up.T)) @ W_down.T
//!
//! All weights stored as [out, in] (PyTorch convention); transposed
//! inside via `vk_transpose_2d` so we can use the F32 matmul kernel
//! that expects [M, K] @ [K, N].
//!
//! Shapes:
//!   x:       [rows, hidden]
//!   W_gate:  [intermediate, hidden]    (becomes [hidden, intermediate] after transpose)
//!   W_up:    [intermediate, hidden]
//!   W_down:  [hidden, intermediate]    (becomes [intermediate, hidden])
//!   out:     [rows, hidden]

use crate::vk_ops::elementwise::vk_mul;
use crate::vk_ops::matmul::vk_matmul;
use crate::vk_ops::shape::vk_transpose_2d;
use crate::vk_ops::silu::vk_silu;
use crate::vk_tensor::{VkDType, VkTensor};
use anyhow::Result;

pub fn vk_swiglu_mlp(
    x: &VkTensor,
    w_gate: &VkTensor,
    w_up: &VkTensor,
    w_down: &VkTensor,
) -> Result<VkTensor> {
    anyhow::ensure!(
        x.shape().len() == 2,
        "vk_swiglu_mlp: x must be rank-2 [rows, hidden]"
    );
    anyhow::ensure!(
        x.dtype() == VkDType::F32
            && w_gate.dtype() == VkDType::F32
            && w_up.dtype() == VkDType::F32
            && w_down.dtype() == VkDType::F32,
        "vk_swiglu_mlp: F32-only"
    );
    let rows = x.shape()[0];
    let hidden = x.shape()[1];
    let intermediate = w_gate.shape()[0];
    anyhow::ensure!(
        w_gate.shape() == [intermediate, hidden],
        "w_gate shape {:?}",
        w_gate.shape()
    );
    anyhow::ensure!(
        w_up.shape() == [intermediate, hidden],
        "w_up shape {:?}",
        w_up.shape()
    );
    anyhow::ensure!(
        w_down.shape() == [hidden, intermediate],
        "w_down shape {:?}",
        w_down.shape()
    );

    // gate = x @ W_gate.T  → [rows, intermediate]
    let w_gate_t = vk_transpose_2d(w_gate)?;
    let gate = vk_matmul(x, &w_gate_t)?;
    let _ = rows;

    // up = x @ W_up.T  → [rows, intermediate]
    let w_up_t = vk_transpose_2d(w_up)?;
    let up = vk_matmul(x, &w_up_t)?;

    // silu(gate)
    let silu_gate = vk_silu(&gate)?;

    // gated = silu_gate * up
    let gated = vk_mul(&silu_gate, &up)?;

    // out = gated @ W_down.T  → [rows, hidden]
    let w_down_t = vk_transpose_2d(w_down)?;
    vk_matmul(&gated, &w_down_t)
}
