use anyhow::Result;
use half::bf16;
use kiln_vulkan_kernel::{CommandBatch, VulkanBuffer, VulkanDevice, Workgroups, kernels, shaders};

fn cpu_linear(x: &[f32], w: &[bf16], batch: usize, hidden: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_dim];
    for row in 0..batch {
        for col in 0..out_dim {
            let mut sum = 0.0f32;
            for h in 0..hidden {
                sum += x[row * hidden + h] * w[h * out_dim + col].to_f32();
            }
            out[row * out_dim + col] = sum;
        }
    }
    out
}

fn append_projection(
    out: &mut [f32],
    offset: usize,
    projection: &[f32],
    batch: usize,
    dim: usize,
) {
    for row in 0..batch {
        let dst = offset + row * dim;
        let src = row * dim;
        out[dst..dst + dim].copy_from_slice(&projection[src..src + dim]);
    }
}

fn run_gdn_in_proj_case(shader: &'static str, row_group_size: usize, batch: usize) -> Result<()> {
    let Ok(dev) = VulkanDevice::new() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let hidden = 7usize;
    let qkv_dim = 13usize;
    let z_dim = 9usize;
    let a_dim = 5usize;
    let b_dim = 3usize;
    let total_out = qkv_dim + z_dim + a_dim + b_dim;
    let x: Vec<f32> = (0..batch * hidden)
        .map(|i| (((i * 17 + 5) % 97) as f32 - 48.0) * 0.015625)
        .collect();
    let qkv_w: Vec<bf16> = (0..hidden * qkv_dim)
        .map(|i| bf16::from_f32((((i * 19 + 3) % 109) as f32 - 54.0) * 0.0068359375))
        .collect();
    let z_w: Vec<bf16> = (0..hidden * z_dim)
        .map(|i| bf16::from_f32((((i * 23 + 7) % 113) as f32 - 56.0) * 0.005859375))
        .collect();
    let a_w: Vec<bf16> = (0..hidden * a_dim)
        .map(|i| bf16::from_f32((((i * 29 + 11) % 127) as f32 - 63.0) * 0.0048828125))
        .collect();
    let b_w: Vec<bf16> = (0..hidden * b_dim)
        .map(|i| bf16::from_f32((((i * 31 + 13) % 131) as f32 - 65.0) * 0.00439453125))
        .collect();

    let x_buf = kernels::upload_f32_buffer_from_slice(&dev, &x)?;
    let qkv_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &qkv_w)?;
    let z_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &z_w)?;
    let a_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &a_w)?;
    let b_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &b_w)?;
    let out_buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (batch * total_out * 4) as u64,
    )?;

    let dispatch_cols = qkv_dim.div_ceil(2) + z_dim.div_ceil(2) + a_dim + b_dim;
    let mut commands = CommandBatch::new(&dev)?;
    commands.record_shader(
        shader,
        &[
            x_buf.handle(),
            qkv_buf.handle(),
            z_buf.handle(),
            a_buf.handle(),
            b_buf.handle(),
            out_buf.handle(),
        ],
        &[
            hidden as u32,
            qkv_dim as u32,
            z_dim as u32,
            a_dim as u32,
            b_dim as u32,
            total_out as u32,
            batch as u32,
        ],
        Workgroups::OneD((batch.div_ceil(row_group_size) * dispatch_cols.div_ceil(80)) as u32),
    )?;
    commands.submit_and_wait("gdn in-proj rows parity")?;

    let bytes = VulkanBuffer::read_back(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        &out_buf,
    )?;
    let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();

    let qkv = cpu_linear(&x, &qkv_w, batch, hidden, qkv_dim);
    let z = cpu_linear(&x, &z_w, batch, hidden, z_dim);
    let a = cpu_linear(&x, &a_w, batch, hidden, a_dim);
    let b = cpu_linear(&x, &b_w, batch, hidden, b_dim);
    let mut expected = vec![0.0f32; batch * total_out];
    append_projection(&mut expected, 0, &qkv, batch, qkv_dim);
    append_projection(&mut expected, batch * qkv_dim, &z, batch, z_dim);
    append_projection(
        &mut expected,
        batch * (qkv_dim + z_dim),
        &a,
        batch,
        a_dim,
    );
    append_projection(
        &mut expected,
        batch * (qkv_dim + z_dim + a_dim),
        &b,
        batch,
        b_dim,
    );

    let max_abs = got
        .iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 1e-4, "max abs diff {max_abs}");
    Ok(())
}

#[test]
fn gdn_in_proj_rows2_matches_cpu_with_tail_rows_and_odd_pairs() -> Result<()> {
    run_gdn_in_proj_case(
        shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS2_BF16W,
        2,
        5,
    )
}

#[test]
fn gdn_in_proj_rows4_matches_cpu_with_tail_rows_and_odd_pairs() -> Result<()> {
    run_gdn_in_proj_case(
        shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS4_BF16W,
        4,
        17,
    )
}

#[test]
fn gdn_in_proj_rows8_matches_cpu_with_tail_rows_and_odd_pairs() -> Result<()> {
    run_gdn_in_proj_case(
        shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS8_BF16W,
        8,
        65,
    )
}
