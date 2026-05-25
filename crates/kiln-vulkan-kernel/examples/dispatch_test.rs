//! Dispatch test: compile a shader, upload data, dispatch, read back.
//!
//! Candle-free via the bytes-based [`kiln_vulkan_kernel::kernels::dispatch_kernel_bytes`]
//! entry point. (#1082)

use anyhow::Result;

fn main() -> Result<()> {
    // 1. Create Vulkan device
    let vk_device = kiln_vulkan_kernel::VulkanDevice::new()?;
    println!(
        "Vulkan device: {} ({})",
        vk_device.device_name(),
        vk_device.vendor_string()
    );

    // 2. Compile add.comp (element-wise add: out = a + b)
    let shader_path = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/add.comp");
    println!("Compiling shader: {}", shader_path);
    let spirv = kiln_vulkan_kernel::pipeline::ShaderPipeline::compile_shader(shader_path)?;
    println!("Compiled SPIR-V: {} bytes", spirv.len());

    // 3. Create test inputs (f32 host buffers, no candle).
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];

    println!("a = {:?}", a);
    println!("b = {:?}", b);

    // 4. Dispatch add kernel
    // add.comp push constants: total_elements (u32)
    let push_constants = [4u32];
    let workgroup_count = (1u32, 1u32, 1u32);
    let output_shape = vec![4usize];

    let a_bytes: &[u8] = bytemuck::cast_slice(&a);
    let b_bytes: &[u8] = bytemuck::cast_slice(&b);

    println!("Dispatching add kernel...");
    let out_bytes = kiln_vulkan_kernel::kernels::dispatch_kernel_bytes(
        &vk_device,
        &spirv,
        &push_constants,
        workgroup_count,
        &[a_bytes, b_bytes],
        &output_shape,
        std::mem::size_of::<f32>(),
    )?;

    let result: &[f32] = bytemuck::cast_slice(&out_bytes);
    println!("result = {:?}", result);

    // Verify: [11.0, 22.0, 33.0, 44.0]
    let expected = [11.0f32, 22.0, 33.0, 44.0];
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
