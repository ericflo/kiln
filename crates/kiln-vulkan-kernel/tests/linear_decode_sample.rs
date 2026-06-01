use anyhow::Result;
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
