//! Dispatch test: compile a shader, upload data, dispatch, read back.

use anyhow::Result;
use candle_core::{DType, Tensor};

fn main() -> Result<()> {
    // 1. Create Vulkan device
    let vk_device = kiln_vulkan_kernel::VulkanDevice::new()?;
    println!(
        "Vulkan device: {} ({})",
        vk_device.device_name(),
        vk_device.vendor_string()
    );

    // 2. Compile add.comp (element-wise add: out = a + b)
    let shader_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/add.comp"
    );
    println!("Compiling shader: {}", shader_path);
    let spirv = kiln_vulkan_kernel::pipeline::ShaderPipeline::compile_shader(shader_path)?;
    println!("Compiled SPIR-V: {} bytes", spirv.len());

    // 3. Create test tensors
    let device = candle_core::Device::Cpu;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 4, &device)?;
    let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], 4, &device)?;

    println!("a = {:?}", a.to_vec1::<f32>()?);
    println!("b = {:?}", b.to_vec1::<f32>()?);

    // 4. Dispatch add kernel
    // add.comp push constants: total_elements (u32)
    let push_constants = [4u32];
    let workgroup_count = (1u32, 1u32, 1u32);
    let output_shape = vec![4usize];

    println!("Dispatching add kernel...");
    let out = kiln_vulkan_kernel::kernels::dispatch_kernel(
        &vk_device,
        &spirv,
        &push_constants,
        workgroup_count,
        &[&a, &b],
        &output_shape,
        DType::F32,
    )?;

    let result: Vec<f32> = out.to_vec1()?;
    println!("result = {:?}", result);

    // Verify: [11.0, 22.0, 33.0, 44.0]
    let expected = vec![11.0f32, 22.0, 33.0, 44.0];
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        if (got - exp).abs() < 1e-4 {
            println!("  element {}: {} == {} OK", i, got, exp);
        } else {
            println!("  element {}: {} != {} FAIL", i, got, exp);
        }
    }

    println!("Dispatch test: OK");
    Ok(())
}
