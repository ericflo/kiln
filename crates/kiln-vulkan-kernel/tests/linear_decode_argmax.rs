use anyhow::Result;
use half::bf16;
use kiln_vulkan_kernel::{VulkanDevice, kernels};

fn expected_argmax(
    x: &[f32],
    weight_t: &[bf16],
    batch: usize,
    hidden: usize,
    out_dim: usize,
) -> Vec<u32> {
    let mut expected = vec![0u32; batch];
    for row in 0..batch {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0u32;
        for col in 0..out_dim {
            let mut score = 0.0f32;
            for h in 0..hidden {
                score += x[row * hidden + h] * weight_t[h * out_dim + col].to_f32();
            }
            if score > best_score || (score == best_score && (col as u32) < best_idx) {
                best_score = score;
                best_idx = col as u32;
            }
        }
        expected[row] = best_idx;
    }
    expected
}

#[test]
fn batched_bf16_argmax_rows4_matches_cpu_with_tail_rows() -> Result<()> {
    let Ok(dev) = VulkanDevice::new() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let batch = 17usize;
    let hidden = 13usize;
    let out_dim = 139usize;
    let x: Vec<f32> = (0..batch * hidden)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.03125)
        .collect();
    let weight_t: Vec<bf16> = (0..hidden * out_dim)
        .map(|i| bf16::from_f32(((i % 31) as f32 - 15.0) * 0.0078125))
        .collect();
    let expected = expected_argmax(&x, &weight_t, batch, hidden, out_dim);
    let weight_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &weight_t)?;

    let got = kernels::dispatch_linear_decode_argmax_batched_cached_bf16_weights_bytes(
        &dev,
        bytemuck::cast_slice(&x),
        &weight_buf,
        batch,
        hidden,
        out_dim,
    )?;
    assert_eq!(got, expected);
    Ok(())
}
