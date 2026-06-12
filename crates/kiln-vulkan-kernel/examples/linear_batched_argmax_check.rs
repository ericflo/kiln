//! Verification probe for batched BF16 linear argmax.

use anyhow::{Context, Result, ensure};
use half::bf16;
use kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_batched_cached_bf16_weights_bytes;
use kiln_vulkan_kernel::{VulkanBuffer, VulkanDevice};

fn upload_u32_buf(dev: &VulkanDevice, data: &[u32]) -> Result<VulkanBuffer> {
    let bytes = bytemuck::cast_slice(data);
    let buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        bytes.len().max(4) as u64,
    )?;
    VulkanBuffer::upload_data(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        &buf,
        bytes,
    )?;
    Ok(buf)
}

fn pack_bf16_pairs(values: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let rounded: Vec<f32> = values.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let mut packed = Vec::with_capacity(values.len().div_ceil(2));
    for pair in rounded.chunks(2) {
        let lo = bf16::from_f32(pair[0]).to_bits() as u32;
        let hi = if pair.len() > 1 {
            (bf16::from_f32(pair[1]).to_bits() as u32) << 16
        } else {
            0
        };
        packed.push(lo | hi);
    }
    (packed, rounded)
}

fn main() -> Result<()> {
    let dev = VulkanDevice::new().context("create Vulkan device")?;
    let batch = 4usize;
    let hidden = 13usize;
    let out_dim = 139usize;

    let x: Vec<f32> = (0..batch * hidden)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.03125)
        .collect();
    let w_f32: Vec<f32> = (0..hidden * out_dim)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.0078125)
        .collect();
    let (w_packed, w_rounded) = pack_bf16_pairs(&w_f32);

    let mut expected = vec![0u32; batch];
    for b in 0..batch {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0u32;
        for c in 0..out_dim {
            let mut score = 0.0f32;
            for h in 0..hidden {
                score += x[b * hidden + h] * w_rounded[h * out_dim + c];
            }
            if score > best_score || (score == best_score && (c as u32) < best_idx) {
                best_score = score;
                best_idx = c as u32;
            }
        }
        expected[b] = best_idx;
    }

    let w_buf = upload_u32_buf(&dev, &w_packed)?;
    let got = dispatch_linear_decode_argmax_batched_cached_bf16_weights_bytes(
        &dev,
        bytemuck::cast_slice(&x),
        &w_buf,
        batch,
        hidden,
        out_dim,
    )?;

    ensure!(
        got == expected,
        "argmax mismatch: got {:?}, expected {:?}",
        got,
        expected
    );
    println!(
        "linear_batched_argmax_check: OK batch={batch} hidden={hidden} out_dim={out_dim} tokens={got:?}"
    );
    Ok(())
}
