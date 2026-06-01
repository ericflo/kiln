use anyhow::Result;
use half::bf16;
use kiln_vulkan_kernel::{VulkanDevice, kernels};

fn f32_bytes(values: &[f32]) -> &[u8] {
    bytemuck::cast_slice(values)
}

#[test]
fn linear_decode_sample_command_batch_returns_argmax_top1() -> Result<()> {
    let Ok(dev) = VulkanDevice::new() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let hidden = 3usize;
    let out_dim = 5usize;
    let x = [1.0f32, -2.0, 0.5];
    let weight_t = [
        1.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, -1.0, 0.0, 2.0, 0.0, //
        0.0, 0.0, 3.0, 0.0, -1.0,
    ];
    let weight_buf = kernels::upload_f32_buffer_from_slice(&dev, &weight_t)?;

    let token = kernels::dispatch_linear_decode_sample_bytes(
        &dev,
        f32_bytes(&x),
        &weight_buf,
        false,
        hidden,
        out_dim,
        &[],
        &[],
        1.0,
        0.0,
        0.0,
        1.0,
        1,
        1.0,
        0.0,
        1234,
    )?;
    assert_eq!(token, 1);
    Ok(())
}

#[test]
fn linear_decode_sample_command_batch_applies_penalties() -> Result<()> {
    let Ok(dev) = VulkanDevice::new() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let hidden = 3usize;
    let out_dim = 5usize;
    let x = [1.0f32, -2.0, 0.5];
    let weight_t = [
        1.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, -1.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 3.0, 3.6, -1.0,
    ];
    let weight_buf = kernels::upload_f32_buffer_from_slice(&dev, &weight_t)?;

    let token = kernels::dispatch_linear_decode_sample_bytes(
        &dev,
        f32_bytes(&x),
        &weight_buf,
        false,
        hidden,
        out_dim,
        &[1],
        &[1],
        1.0,
        10.0,
        0.0,
        1.0,
        1,
        1.0,
        0.0,
        1234,
    )?;
    assert_eq!(token, 3);
    Ok(())
}

#[test]
fn linear_decode_sample_batch_returns_argmax_top1() -> Result<()> {
    let Ok(dev) = VulkanDevice::new() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let batch = 2usize;
    let hidden = 3usize;
    let out_dim = 5usize;
    let x = [
        1.0f32, -2.0, 0.5, //
        0.0, 1.0, 2.0,
    ];
    let weight_t = [
        1.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, -1.0, 0.0, 2.0, 0.0, //
        0.0, 0.0, 3.0, 0.0, -1.0,
    ];
    let weight_buf = kernels::upload_f32_buffer_from_slice(&dev, &weight_t)?;

    let tokens = kernels::dispatch_linear_decode_sample_batch_bytes(
        &dev,
        f32_bytes(&x),
        &weight_buf,
        false,
        batch,
        hidden,
        out_dim,
        &[],
        &[],
        &[],
        &[1.0, 1.0],
        &[0.0, 0.0],
        &[0.0, 0.0],
        &[1.0, 1.0],
        &[1, 1],
        &[1.0, 1.0],
        &[0.0, 0.0],
        &[1234, 5678],
    )?;
    assert_eq!(tokens, vec![1, 2]);
    Ok(())
}

#[test]
fn linear_decode_sample_batch_applies_row_penalties() -> Result<()> {
    let Ok(dev) = VulkanDevice::new() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let batch = 2usize;
    let hidden = 3usize;
    let out_dim = 5usize;
    let x = [
        1.0f32, -2.0, 0.5, //
        0.0, 1.0, 2.0,
    ];
    let weight_t = [
        1.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, -1.0, 3.0, 0.0, 0.0, //
        0.0, 0.0, 3.0, 3.6, -1.0,
    ];
    let weight_buf = kernels::upload_f32_buffer_from_slice(&dev, &weight_t)?;

    let tokens = kernels::dispatch_linear_decode_sample_batch_bytes(
        &dev,
        f32_bytes(&x),
        &weight_buf,
        false,
        batch,
        hidden,
        out_dim,
        &[0],
        &[1],
        &[1],
        &[1.0, 1.0],
        &[10.0, 0.0],
        &[0.0, 0.0],
        &[1.0, 1.0],
        &[1, 1],
        &[1.0, 1.0],
        &[0.0, 0.0],
        &[1234, 5678],
    )?;
    assert_eq!(tokens, vec![3, 2]);
    Ok(())
}

#[test]
fn linear_decode_sample_batch_rows8_bf16_top1_matches_cpu() -> Result<()> {
    let Ok(dev) = VulkanDevice::new() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let batch = 65usize;
    let hidden = 11usize;
    let out_dim = 67usize;
    let x: Vec<f32> = (0..batch * hidden)
        .map(|i| (((i * 17 + 13) % 101) as f32 - 50.0) * 0.013671875)
        .collect();
    let weight_t: Vec<bf16> = (0..hidden * out_dim)
        .map(|i| bf16::from_f32((((i * 31 + 7) % 127) as f32 - 63.0) * 0.0048828125))
        .collect();
    let weight_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &weight_t)?;

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

    let tokens = kernels::dispatch_linear_decode_sample_batch_bytes(
        &dev,
        f32_bytes(&x),
        &weight_buf,
        true,
        batch,
        hidden,
        out_dim,
        &[],
        &[],
        &[],
        &vec![1.0; batch],
        &vec![0.0; batch],
        &vec![0.0; batch],
        &vec![1.0; batch],
        &vec![1; batch],
        &vec![1.0; batch],
        &vec![0.0; batch],
        &vec![1234; batch],
    )?;
    assert_eq!(tokens, expected);
    Ok(())
}
