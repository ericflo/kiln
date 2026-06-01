//! Verification probe for split-K paged decode attention.

use anyhow::{ensure, Context, Result};
use kiln_vulkan_kernel::kernels::{
    dispatch_paged_attn_decode_batch_paged_f32_bytes,
    dispatch_paged_attn_decode_batch_paged_splitk_f32_bytes, paged_attn_decode_splitk_chunks,
};
use kiln_vulkan_kernel::VulkanDevice;

fn patterned_f32(len: usize, period: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..len)
        .map(|i| bias + ((i % period) as f32 - (period / 2) as f32) * scale)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() -> Result<()> {
    let dev = VulkanDevice::new().context("create Vulkan device")?;
    let batch = 3usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 16usize;
    let total_slots = 12usize;
    let max_blocks_per_seq = 3usize;
    let page_block_size = 4usize;
    let seq_lens = [7u32, 5, 9];
    let block_table = [2u32, 0, 1, 1, 2, 0, 0, 1, 2];
    let softmax_scale = (head_dim as f32).sqrt().recip();
    let num_chunks = paged_attn_decode_splitk_chunks(batch, max_blocks_per_seq);

    let q = patterned_f32(batch * num_heads * head_dim, 17, 0.015, 0.01);
    let k_pool = patterned_f32(total_slots * num_kv_heads * head_dim, 23, 0.012, -0.005);
    let v_pool = patterned_f32(total_slots * num_kv_heads * head_dim, 29, 0.02, 0.02);

    let base = dispatch_paged_attn_decode_batch_paged_f32_bytes(
        &dev,
        bytemuck::cast_slice(&q),
        bytemuck::cast_slice(&k_pool),
        bytemuck::cast_slice(&v_pool),
        batch,
        num_heads,
        head_dim,
        total_slots,
        num_kv_heads,
        &block_table,
        &seq_lens,
        max_blocks_per_seq,
        page_block_size,
        softmax_scale,
    )
    .context("run non-split paged attention")?;
    let split = dispatch_paged_attn_decode_batch_paged_splitk_f32_bytes(
        &dev,
        bytemuck::cast_slice(&q),
        bytemuck::cast_slice(&k_pool),
        bytemuck::cast_slice(&v_pool),
        batch,
        num_heads,
        head_dim,
        total_slots,
        num_kv_heads,
        &block_table,
        &seq_lens,
        max_blocks_per_seq,
        page_block_size,
        softmax_scale,
        num_chunks,
    )
    .context("run split-K paged attention")?;

    let base_f32: &[f32] = bytemuck::cast_slice(&base);
    let split_f32: &[f32] = bytemuck::cast_slice(&split);
    let max_abs = max_abs_diff(base_f32, split_f32);
    ensure!(
        max_abs <= 1.0e-5,
        "split-K paged attention mismatch: max_abs={max_abs:.6e}"
    );
    println!(
        "paged_attn_splitk_check: OK batch={batch} heads={num_heads} kv_heads={num_kv_heads} head_dim={head_dim} chunks={num_chunks} max_abs={max_abs:.6e}"
    );
    Ok(())
}
