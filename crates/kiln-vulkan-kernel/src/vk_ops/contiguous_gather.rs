//! On-device strided gather → contiguous (F32).
//!
//! `vk_gather_contiguous_f32` materializes an arbitrary strided/offset view
//! of a buffer into a fresh, packed C-contiguous `VkTensor` — the resident
//! replacement for kt's `Tensor::contiguous()` host bounce (D2H per-element
//! gather + H2D re-upload). Every `transpose().contiguous()`,
//! `narrow().contiguous()`, GQA-expand, etc. on a Vulkan tensor went through
//! that bounce; this keeps it on the GPU.
//!
//! The view is described by raw layout metadata (whole source buffer + shape
//! + element strides + element start_offset) rather than a `VkTensor`,
//! because a `VkTensor` is by construction whole-buffer/contiguous and cannot
//! represent the strided input.

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::Result;
use std::sync::Arc;

/// Maximum logical rank the gather shader's fixed push-constant arrays cover.
/// Mirrors the `shape[8]` / `strides[8]` declarations in
/// `csrc/shaders/vk_gather_contiguous_f32.comp`.
pub const MAX_RANK: usize = 8;

/// Gather the strided F32 view `(src, shape, strides, start_offset)` (all
/// extents/strides/offset in ELEMENTS) into a fresh contiguous F32
/// `VkTensor` of `shape`.
///
/// `src` is the *whole* source buffer; `start_offset` + `strides` locate each
/// logical element within it. The output is densely packed in row-major
/// order.
///
/// # Errors
///
/// Returns an error if `rank > MAX_RANK`, the `shape`/`strides` ranks differ,
/// or the dispatch fails.
pub fn vk_gather_contiguous_f32(
    device: &Arc<VulkanDevice>,
    src: &Arc<VulkanBuffer>,
    shape: &[usize],
    strides: &[usize],
    start_offset: usize,
) -> Result<VkTensor> {
    let rank = shape.len();
    anyhow::ensure!(
        rank <= MAX_RANK,
        "vk_gather_contiguous_f32: rank {rank} exceeds MAX_RANK {MAX_RANK}"
    );
    anyhow::ensure!(
        strides.len() == rank,
        "vk_gather_contiguous_f32: shape rank {rank} != strides rank {}",
        strides.len()
    );

    // Element count: product of extents (empty product == 1 for rank-0
    // scalars; any zero extent yields 0 → an empty output).
    let n: usize = shape.iter().product();
    // Allocate at least one element so the buffer handle is valid even for an
    // empty result (the dispatch is skipped in that case).
    let out = crate::buffer_pool::pool_alloc_f32(device, n.max(1))?;

    // push = [rank, n, start_offset, shape[8], strides[8]]
    let mut push: Vec<u32> = Vec::with_capacity(3 + 2 * MAX_RANK);
    push.push(rank as u32);
    push.push(n as u32);
    push.push(start_offset as u32);
    let mut sh = [0u32; MAX_RANK];
    let mut st = [0u32; MAX_RANK];
    for ax in 0..rank {
        sh[ax] = shape[ax] as u32;
        st[ax] = strides[ax] as u32;
    }
    push.extend_from_slice(&sh);
    push.extend_from_slice(&st);

    if n > 0 {
        let workgroups = ((n + 255) / 256) as u32;
        dispatch_simple(
            device,
            "vk_gather_contiguous_f32",
            &[src.handle(), out.handle()],
            &push,
            workgroups,
        )?;
    }

    Ok(VkTensor::from_buffer(
        out,
        shape.to_vec(),
        VkDType::F32,
        Arc::clone(device),
    ))
}
