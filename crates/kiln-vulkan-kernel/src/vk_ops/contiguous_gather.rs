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
use anyhow::{Context, Result};
use std::sync::Arc;

/// Maximum logical rank the gather shader's fixed push-constant arrays cover.
/// Mirrors the `shape[8]` / `strides[8]` declarations in
/// `csrc/shaders/vk_gather_contiguous_f32.comp`.
pub const MAX_RANK: usize = 8;

/// Build the `[rank, n, start_offset, shape[8], strides[8]]` push-constant
/// vector shared by the F32 and BF16 gather shaders.
fn gather_push(rank: usize, n: usize, start_offset: usize, shape: &[usize], strides: &[usize]) -> Vec<u32> {
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
    push
}

fn check_gather_ranks(shape: &[usize], strides: &[usize]) -> Result<usize> {
    let rank = shape.len();
    anyhow::ensure!(
        rank <= MAX_RANK,
        "vk_gather_contiguous: rank {rank} exceeds MAX_RANK {MAX_RANK}"
    );
    anyhow::ensure!(
        strides.len() == rank,
        "vk_gather_contiguous: shape rank {rank} != strides rank {}",
        strides.len()
    );
    Ok(rank)
}

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
    let rank = check_gather_ranks(shape, strides)?;
    // Element count: product of extents (empty product == 1 for rank-0
    // scalars; any zero extent yields 0 → an empty output).
    let n: usize = shape.iter().product();
    // Allocate at least one element so the buffer handle is valid even for an
    // empty result (the dispatch is skipped in that case).
    let out = crate::buffer_pool::pool_alloc_f32(device, n.max(1))?;
    let push = gather_push(rank, n, start_offset, shape, strides);

    if n > 0 {
        // One invocation per output element.
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

/// BF16 twin of [`vk_gather_contiguous_f32`]. BF16 is stored as u16 elements
/// packed into u32 words (low/high lane); the shader writes one output word
/// per invocation (two logical elements), so there are no inter-lane races —
/// same convention as `vk_transpose_2d_bf16`.
///
/// # Errors
///
/// Returns an error if `rank > MAX_RANK`, the `shape`/`strides` ranks differ,
/// the output buffer cannot be allocated, or the dispatch fails.
pub fn vk_gather_contiguous_bf16(
    device: &Arc<VulkanDevice>,
    src: &Arc<VulkanBuffer>,
    shape: &[usize],
    strides: &[usize],
    start_offset: usize,
) -> Result<VkTensor> {
    let rank = check_gather_ranks(shape, strides)?;
    let n: usize = shape.iter().product();

    // BF16 output: n * 2 bytes, rounded up to a u32-word multiple (the shader
    // addresses storage as u32-packed pairs). Mirrors `alloc_transpose_output`.
    let bytes = (((n * 2).max(2) + 3) / 4) * 4;
    let out = Arc::new(
        VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            bytes as u64,
        )
        .context("vk_gather_contiguous_bf16: alloc output buffer")?,
    );
    let push = gather_push(rank, n, start_offset, shape, strides);

    if n > 0 {
        // One invocation per output WORD (two packed BF16 elements).
        let total_words = (n + 1) / 2;
        let workgroups = ((total_words + 255) / 256) as u32;
        dispatch_simple(
            device,
            "vk_gather_contiguous_bf16",
            &[src.handle(), out.handle()],
            &push,
            workgroups,
        )?;
    }

    Ok(VkTensor::from_buffer(
        out,
        shape.to_vec(),
        VkDType::Bf16,
        Arc::clone(device),
    ))
}
