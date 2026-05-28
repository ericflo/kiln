//! Candle ↔ Vulkan tensor bridge.
//!
//! This module is the **explicit candle bridge seam** for
//! `kiln-vulkan-kernel`. Every public fn here takes or returns a
//! `candle_core::Tensor`; everything else in the crate operates on raw
//! `&[u8]` / `Vec<u8>` plus shape metadata.
//!
//! Issue #1082 tracks the full candle removal. Until the upstream
//! callers (`kiln-model` weight loading, `vulkan_lora_op`, the
//! `gdn_parity` test) finish migrating to candle-free APIs, this module
//! is the **only** place in `kiln-vulkan-kernel` where `Tensor` appears
//! at the public surface.
//!
//! Migration policy (chokepoint pattern, mirrors `kiln-tensor::metal_types`):
//!
//! * `kernels.rs` is candle-free at the public-surface level. The 6
//!   bridge fns it used to expose are re-exported from here for
//!   back-compat, so existing call sites continue to work unchanged.
//! * Internal `kernels.rs` plumbing that needs candle staging (e.g.
//!   the `*_bytes` dispatch shims that build a CPU `Tensor` from raw
//!   bytes and then call into an older `_impl`) reach into this module
//!   via `crate::candle_bridge::*` rather than redefining helpers.
//! * New callers MUST NOT add candle pub fns to `kernels.rs`. Add them
//!   here, or — preferably — implement a candle-free counterpart that
//!   takes `&[u8]` / `&[f32]` / `&[bf16]` directly.
//!
//! When the candle removal completes, this whole file deletes; the
//! `pub use crate::candle_bridge::*;` line in `kernels.rs` deletes; and
//! `kiln-vulkan-kernel` becomes fully candle-free. (#1082)

use crate::buffer::VulkanBuffer;
use crate::device::VulkanDevice;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use half::bf16;

/// Extract raw f32 bytes from a candle-core Tensor.
pub fn extract_tensor_bytes(tensor: &Tensor) -> Result<(Vec<u8>, Vec<usize>)> {
    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    let flat = tensor.flatten_all().context("failed to flatten tensor")?;
    let f32_data = flat
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()
        .context("failed to extract f32 data")?;
    Ok((bytemuck::cast_slice(&f32_data).to_vec(), shape))
}

/// Extract raw bf16 weights packed two values per u32 in row-major order.
///
/// Shaders expand each 16-bit lane with `uintBitsToFloat(bits << 16)`, which
/// preserves the exact bf16 value without requiring native shader bf16 support.
/// Public re-export for kiln-model's residency registry — same impl
/// as the private [`extract_tensor_packed_bf16_bytes`]. Used by the
/// `register_resident_activation` BF16 path to upload bytes in the
/// layout every Vulkan kernel's `load_weight` helper expects.
pub fn extract_tensor_packed_bf16_bytes_pub(tensor: &Tensor) -> Result<(Vec<u8>, Vec<usize>)> {
    extract_tensor_packed_bf16_bytes(tensor)
}

/// Crate-internal helper: extract bf16 weights packed two-per-u32. Mirrors the
/// shape semantics of `extract_tensor_packed_bf16_bytes_pub` but is the impl
/// reached by `*_bf16w_*_bytes` dispatch shims inside this crate. (#1082)
pub(crate) fn extract_tensor_packed_bf16_bytes(
    tensor: &Tensor,
) -> Result<(Vec<u8>, Vec<usize>)> {
    anyhow::ensure!(
        tensor.dtype() == DType::BF16,
        "packed bf16 upload requires BF16 tensor, got {:?}",
        tensor.dtype()
    );
    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    let flat = tensor.flatten_all().context("failed to flatten tensor")?;
    let bf16_data = flat
        .to_vec1::<bf16>()
        .context("failed to extract bf16 data")?;
    let mut packed = Vec::with_capacity(bf16_data.len().div_ceil(2));
    for pair in bf16_data.chunks(2) {
        let lo = pair[0].to_bits() as u32;
        let hi = pair.get(1).map(|v| v.to_bits() as u32).unwrap_or(0);
        packed.push(lo | (hi << 16));
    }
    Ok((bytemuck::cast_slice(&packed).to_vec(), shape))
}

/// Create a candle-core Tensor from raw bytes.
pub fn create_tensor_from_data(data: &[u8], shape: &[usize], dtype: DType) -> Result<Tensor> {
    let f32_data: &[f32] = bytemuck::cast_slice(data);
    let tensor =
        Tensor::from_vec(f32_data.to_vec(), f32_data.len(), &Device::Cpu)?.reshape(shape)?;

    if dtype == DType::BF16 {
        Ok(tensor.to_dtype(DType::BF16)?)
    } else {
        Ok(tensor)
    }
}

/// Build a CPU F32 candle Tensor from raw f32 bytes plus shape.
///
/// Historically used internally by the `*_bytes` dispatch shims so
/// callers could stay candle-free even when the underlying `_impl`
/// still needed a `Tensor`. After PRs 9be4971b, a8167473, 2fc8e347,
/// and the gdn_in_proj inversion, all `_impl` dispatchers in `kernels.rs`
/// take `&[u8]` directly, so this staging helper currently has no
/// callers. Kept (with a dead-code allow) alongside the BF16 sibling
/// so future `_bytes` shims that re-introduce a candle staging step
/// have a drop-in helper without churning the bridge surface. (#1082)
#[allow(dead_code)]
pub(crate) fn build_cpu_f32_tensor_from_bytes(
    data: &[u8],
    shape: &[usize],
) -> Result<Tensor> {
    create_tensor_from_data(data, shape, DType::F32)
}

/// Build a CPU BF16 candle Tensor from packed bf16 bytes plus shape.
/// Used internally by `*_bytes` dispatch shims that still need a
/// candle staging tensor. (#1082)
///
/// Currently dead code — preserved alongside its f32 sibling so the
/// next `*_bytes` shim that needs a BF16 staging tensor has a drop-in
/// helper and we don't churn the bridge surface again. Pre-existing
/// `*_never_used` warning that survived the chokepoint move.
#[allow(dead_code)]
pub(crate) fn build_cpu_bf16_tensor_from_bytes(
    data: &[u8],
    shape: &[usize],
) -> Result<Tensor> {
    let expected: usize = shape.iter().product::<usize>() * 2;
    anyhow::ensure!(
        data.len() == expected,
        "build_cpu_bf16_tensor_from_bytes: expected {expected} bytes for shape {:?}, got {}",
        shape,
        data.len()
    );
    let half_slice: &[half::bf16] = bytemuck::cast_slice(data);
    Tensor::from_slice(half_slice, shape, &Device::Cpu).context("build bf16 cpu tensor")
}

/// Decode a registry-resident `VulkanBuffer` back into a candle CPU
/// Tensor of the requested `shape` and `dtype`.
///
/// Inverse of the encoding choices in
/// `vulkan::register_resident_activation`: BF16 entries are stored as
/// packed bf16 (two bf16 lanes per u32 word, `(hi << 16) | lo`), F32
/// entries are stored as raw f32 bytes. The decoder bit-expands each
/// bf16 lane back to f32 then casts to the target dtype via candle so
/// we don't need a hard dependency on the `half` crate at this layer.
///
/// Used by `VulkanLoraOp::bwd` to read LoRA `A` and `B` weights
/// straight from the registry instead of candle CPU storage —
/// closes the candle-storage staleness gap that the lazy
/// `sync_to_candle` flow opens.
pub fn buffer_to_tensor(
    vk_device: &VulkanDevice,
    buffer: &VulkanBuffer,
    shape: &[usize],
    dtype: DType,
) -> Result<Tensor> {
    let bytes = VulkanBuffer::read_back(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        buffer,
    )
    .context("buffer_to_tensor: VulkanBuffer::read_back")?;
    if dtype == DType::BF16 {
        anyhow::ensure!(
            bytes.len() % 2 == 0,
            "buffer_to_tensor BF16: buffer byte count {} is not a multiple of 2",
            bytes.len()
        );
        let elem_count: usize = shape.iter().product();
        let stored = bytes.len() / 2;
        anyhow::ensure!(
            stored >= elem_count,
            "buffer_to_tensor BF16: buffer holds {} bf16 elements, expected at least {} \
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
        Ok(Tensor::from_vec(f32_data, shape, &Device::Cpu)?.to_dtype(DType::BF16)?)
    } else {
        create_tensor_from_data(&bytes, shape, dtype)
    }
}

/// Upload raw bytes as immutable weights into a device-local Vulkan
/// buffer using a transient command pool. Candle-free core shared by
/// [`upload_tensor_f32_buffer`] and [`upload_tensor_bf16_packed_buffer`]
/// and reusable from `#1082` migration call sites that already have
/// host bytes in hand. (#1082)
///
/// Lives in `candle_bridge` because all current callers reach it
/// through the candle-shim wrappers. The candle-free
/// `upload_f32_buffer_from_slice` / `upload_bf16_packed_buffer_from_slice`
/// pair in `kernels.rs` also delegate here.
pub(crate) fn upload_bytes_to_device_buffer(
    vk_device: &VulkanDevice,
    bytes: &[u8],
    create_ctx: &'static str,
    upload_ctx: &'static str,
) -> Result<VulkanBuffer> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let buffer = VulkanBuffer::create_device_local(device, device_local_mt, bytes.len() as u64)
        .context(create_ctx)?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &buffer,
            bytes,
        )
        .context(upload_ctx)?;
    }
    Ok(buffer)
}

/// Upload a Candle tensor as contiguous f32 values into a device-local Vulkan buffer.
///
/// This is used by model-level caches for immutable weights so repeated decode
/// steps do not re-upload multi-megabyte projection matrices.
///
/// Candle-shim wrapper around the candle-free
/// `kernels::upload_f32_buffer_from_slice` — extracts the tensor's f32
/// bytes once, then delegates to the shared byte-upload core. (#1082)
pub fn upload_tensor_f32_buffer(
    vk_device: &VulkanDevice,
    tensor: &Tensor,
) -> Result<VulkanBuffer> {
    let tensor_f32;
    let tensor = if tensor.dtype() == DType::F32 {
        tensor
    } else {
        tensor_f32 = tensor
            .to_dtype(DType::F32)
            .context("failed to convert cached tensor to f32 for Vulkan upload")?;
        &tensor_f32
    };
    let data = extract_tensor_bytes(tensor)?.0;
    upload_bytes_to_device_buffer(
        vk_device,
        &data,
        "failed to create cached tensor buffer",
        "failed to upload cached tensor buffer",
    )
}

/// Upload a BF16 Candle tensor as packed immutable weights into a Vulkan buffer.
///
/// The resulting buffer stores two BF16 values per u32, matching the
/// `*_bf16w.comp` shader variants.
///
/// Candle-shim wrapper around the candle-free
/// `kernels::upload_bf16_packed_buffer_from_slice`. Extracts packed
/// bf16 bytes from the tensor once, then delegates to the shared
/// byte-upload core. (#1082)
pub fn upload_tensor_bf16_packed_buffer(
    vk_device: &VulkanDevice,
    tensor: &Tensor,
) -> Result<VulkanBuffer> {
    let data = extract_tensor_packed_bf16_bytes(tensor)?.0;
    upload_bytes_to_device_buffer(
        vk_device,
        &data,
        "failed to create cached packed bf16 tensor buffer",
        "failed to upload cached packed bf16 tensor buffer",
    )
}
