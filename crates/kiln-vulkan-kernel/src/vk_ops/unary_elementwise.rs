//! Generic unary elementwise math for `VkTensor` (F32).
//!
//! One Vulkan kernel covers the whole family of elementwise unary ops
//! (neg/exp/ln/sqrt/abs/recip/sign/floor/.../sin/cos/tan/tanh/gelu/relu)
//! plus scalar add/mul, so the kt `DeviceOp1` layer stays fully
//! GPU-resident on Vulkan instead of falling back to the CPU host bounce.
//! The `op` code selects the function; it must stay in sync with the
//! `vk_unary_elementwise_f32.comp` shader switch.

use crate::vk_ops::{dispatch_simple, for_each_1d_tile, vk_exp_tile_elements};
use crate::vk_tensor::{VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::Result;
use std::sync::Arc;

/// Op codes — keep in sync with the shader switch in
/// `csrc/shaders/vk_unary_elementwise_f32.comp`.
pub mod op {
    pub const NEG: u32 = 0;
    pub const EXP: u32 = 1;
    pub const LN: u32 = 2;
    pub const SQRT: u32 = 3;
    pub const ABS: u32 = 4;
    pub const RECIP: u32 = 5;
    pub const SIGN: u32 = 6;
    pub const FLOOR: u32 = 7;
    pub const CEIL: u32 = 8;
    pub const ROUND: u32 = 9;
    pub const TRUNC: u32 = 10;
    pub const SIN: u32 = 11;
    pub const COS: u32 = 12;
    pub const TAN: u32 = 13;
    pub const ADD_SCALAR: u32 = 14;
    pub const TANH: u32 = 15;
    pub const GELU: u32 = 16;
    pub const RELU: u32 = 17;
    pub const EXP2: u32 = 18;
    pub const LOG2: u32 = 19;
    pub const EXPM1: u32 = 20;
    pub const LOG1P: u32 = 21;
    pub const SINH: u32 = 22;
    pub const COSH: u32 = 23;
    pub const ASIN: u32 = 24;
    pub const ACOS: u32 = 25;
    pub const ATAN: u32 = 26;
    pub const MUL_SCALAR: u32 = 27;
    pub const LOG10: u32 = 28;
}

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

fn dispatch_unary(
    device: &VulkanDevice,
    x: &VulkanBuffer,
    out: &VulkanBuffer,
    n: usize,
    op: u32,
    param0: f32,
) -> Result<()> {
    let tile_elements = vk_exp_tile_elements();
    if n <= tile_elements {
        let workgroups = ((n + 255) / 256) as u32;
        let push = [n as u32, op, param0.to_bits()];
        return dispatch_simple(
            device,
            "vk_unary_elementwise_f32",
            &[x.handle(), out.handle()],
            &push,
            workgroups,
        );
    }
    for_each_1d_tile(n, tile_elements, |offset, len| {
        let workgroups = ((len + 255) / 256) as u32;
        let push = [len as u32, offset as u32, op, param0.to_bits()];
        dispatch_simple(
            device,
            "vk_unary_elementwise_f32_offset",
            &[x.handle(), out.handle()],
            &push,
            workgroups,
        )
    })
}

/// Apply a unary elementwise op (`op`, see [`op`]) to every element of
/// `x`, returning a fresh contiguous F32 `VkTensor` of the same shape.
/// `param0` is the scalar operand for [`op::ADD_SCALAR`] /
/// [`op::MUL_SCALAR`] and ignored otherwise.
pub fn vk_unary_elementwise_f32(x: &VkTensor, op: u32, param0: f32) -> Result<VkTensor> {
    anyhow::ensure!(
        x.dtype() == VkDType::F32,
        "vk_unary_elementwise: F32 only (got {:?})",
        x.dtype()
    );
    let n = x.num_elements();
    let out = alloc_f32(x.device(), n)?;
    dispatch_unary(x.device(), x.buffer(), &out, n, op, param0)?;
    Ok(VkTensor::from_buffer(
        out,
        x.shape().to_vec(),
        VkDType::F32,
        Arc::clone(x.device()),
    ))
}
