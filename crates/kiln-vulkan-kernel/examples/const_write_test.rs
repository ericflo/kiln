//! Test constant write to verify output path.

use anyhow::Result;
use candle_core::DType;

fn main() -> Result<()> {
    let vk_device = kiln_vulkan_kernel::VulkanDevice::new()?;
    println!("Device: {}", vk_device.device_name());

    // Compile constant write shader
    let spirv = kiln_vulkan_kernel::pipeline::ShaderPipeline::compile_shader("/tmp/const_write.comp")?;
    println!("Compiled SPIR-V: {} bytes", spirv.len());

    // Dispatch: no inputs, just output
    let push_constants = [4u32]; // 4 elements
    let workgroup_count = (1u32, 1u32, 1u32);
    let output_shape = vec![4usize];

    let out = kiln_vulkan_kernel::kernels::dispatch_kernel(
        &vk_device,
        &spirv,
        &push_constants,
        workgroup_count,
        &[],  // no input tensors
        &output_shape,
        DType::F32,
    )?;

    let result: Vec<f32> = out.to_vec1()?;
    println!("result = {:?}", result);
    
    let all_99 = result.iter().all(|&x| (x - 99.0).abs() < 1e-4);
    if all_99 {
        println!("CONST WRITE TEST: PASS (all 99.0)");
    } else {
        println!("CONST WRITE TEST: FAIL");
    }

    Ok(())
}
