use anyhow::Result;
use kiln_vulkan_kernel::{shaders, CommandBatch, VulkanBuffer, VulkanDevice, Workgroups};

fn upload_f32(dev: &VulkanDevice, data: &[f32]) -> Result<VulkanBuffer> {
    let bytes = bytemuck::cast_slice(data);
    let buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        bytes.len() as u64,
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

fn empty_f32(dev: &VulkanDevice, len: usize) -> Result<VulkanBuffer> {
    VulkanBuffer::create_device_local(dev.device(), dev.device_local_mem_type(), (len * 4) as u64)
}

fn read_f32(dev: &VulkanDevice, buf: &VulkanBuffer) -> Result<Vec<f32>> {
    let bytes = VulkanBuffer::read_back(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        buf,
    )?;
    Ok(bytemuck::cast_slice(&bytes).to_vec())
}

fn assert_close(name: &str, got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "{name} length");
    let max_abs = got
        .iter()
        .zip(expected)
        .map(|(g, e)| (g - e).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs <= tol, "{name} max abs diff {max_abs}");
}

#[test]
fn gdn_qk_norm_recurrent_fusion_matches_two_dispatch_path() -> Result<()> {
    if !VulkanDevice::probe() {
        return Ok(());
    }
    let dev = VulkanDevice::new()?;
    let batch = 3usize;
    let nk = 3usize;
    let gqa_ratio = 2usize;
    let nv = nk * gqa_ratio;
    let dk = 8usize;
    let dv = 8usize;
    let eps = 1e-6f32;
    let l2_eps = 1e-6f32;
    let q_scale = 1.0f32 / (dk as f32).sqrt();
    let k_scale = 1.0f32;

    let q: Vec<f32> = (0..batch * nk * dk)
        .map(|i| ((i % 41) as f32 - 20.0) * 0.017)
        .collect();
    let k: Vec<f32> = (0..batch * nk * dk)
        .map(|i| ((i % 37) as f32 - 18.0) * -0.013)
        .collect();
    let v: Vec<f32> = (0..batch * nv * dv)
        .map(|i| ((i % 43) as f32 - 21.0) * 0.011)
        .collect();
    let a: Vec<f32> = (0..batch * nv)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.023)
        .collect();
    let b: Vec<f32> = (0..batch * nv)
        .map(|i| ((i % 19) as f32 - 9.0) * -0.021)
        .collect();
    let a_log: Vec<f32> = (0..nv).map(|i| -0.17 + i as f32 * 0.019).collect();
    let dt_bias: Vec<f32> = (0..nv).map(|i| -0.03 + i as f32 * 0.007).collect();
    let state: Vec<f32> = (0..batch * nv * dk * dv)
        .map(|i| ((i % 53) as f32 - 26.0) * 0.004)
        .collect();
    let z: Vec<f32> = (0..batch * nv * dv)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.015)
        .collect();
    let weight: Vec<f32> = (0..dv).map(|i| 0.7 + i as f32 * 0.031).collect();

    let q_buf = upload_f32(&dev, &q)?;
    let k_buf = upload_f32(&dev, &k)?;
    let v_buf = upload_f32(&dev, &v)?;
    let a_buf = upload_f32(&dev, &a)?;
    let b_buf = upload_f32(&dev, &b)?;
    let a_log_buf = upload_f32(&dev, &a_log)?;
    let dt_bias_buf = upload_f32(&dev, &dt_bias)?;
    let z_buf = upload_f32(&dev, &z)?;
    let weight_buf = upload_f32(&dev, &weight)?;

    let q_expanded = empty_f32(&dev, batch * nv * dk)?;
    let k_expanded = empty_f32(&dev, batch * nv * dk)?;
    let state_ref = upload_f32(&dev, &state)?;
    let state_fused = upload_f32(&dev, &state)?;
    let out_ref = empty_f32(&dev, batch * nv * dv)?;
    let out_fused = empty_f32(&dev, batch * nv * dv)?;

    let mut reference = CommandBatch::new(&dev)?;
    reference.record_shader(
        shaders::L2_NORM_QK_PER_ROW,
        &[
            q_buf.handle(),
            k_buf.handle(),
            q_expanded.handle(),
            k_expanded.handle(),
        ],
        &[
            (batch * nk) as u32,
            dk as u32,
            l2_eps.to_bits(),
            q_scale.to_bits(),
            k_scale.to_bits(),
            gqa_ratio as u32,
        ],
        Workgroups::OneD((batch * nk) as u32),
    )?;
    reference.record_shader(
        shaders::GDN_DECODE_GATES_RECURRENT_RMSNORM,
        &[
            q_expanded.handle(),
            k_expanded.handle(),
            v_buf.handle(),
            a_buf.handle(),
            b_buf.handle(),
            a_log_buf.handle(),
            dt_bias_buf.handle(),
            state_ref.handle(),
            z_buf.handle(),
            weight_buf.handle(),
            out_ref.handle(),
        ],
        &[nv as u32, dk as u32, dv as u32, eps.to_bits(), batch as u32],
        Workgroups::OneD((batch * nv) as u32),
    )?;
    reference.submit_and_wait("gdn qk norm recurrent reference")?;

    let mut fused = CommandBatch::new(&dev)?;
    fused.record_shader(
        shaders::GDN_DECODE_QK_NORM_GATES_RECURRENT_RMSNORM,
        &[
            q_buf.handle(),
            k_buf.handle(),
            v_buf.handle(),
            a_buf.handle(),
            b_buf.handle(),
            a_log_buf.handle(),
            dt_bias_buf.handle(),
            state_fused.handle(),
            z_buf.handle(),
            weight_buf.handle(),
            out_fused.handle(),
        ],
        &[
            nk as u32,
            dk as u32,
            dv as u32,
            eps.to_bits(),
            batch as u32,
            gqa_ratio as u32,
            l2_eps.to_bits(),
            q_scale.to_bits(),
            k_scale.to_bits(),
        ],
        Workgroups::OneD((batch * nk) as u32),
    )?;
    fused.submit_and_wait("gdn qk norm recurrent fused")?;

    assert_close(
        "out",
        &read_f32(&dev, &out_fused)?,
        &read_f32(&dev, &out_ref)?,
        2e-5,
    );
    assert_close(
        "state",
        &read_f32(&dev, &state_fused)?,
        &read_f32(&dev, &state_ref)?,
        2e-5,
    );
    Ok(())
}
