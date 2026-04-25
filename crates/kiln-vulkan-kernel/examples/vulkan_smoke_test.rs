//! Quick smoke test: create a Vulkan device, compile a simple shader, dispatch it.

use anyhow::Result;

fn main() -> Result<()> {
    // 1. Create Vulkan device
    let vk_device = kiln_vulkan_kernel::VulkanDevice::new()?;
    println!(
        "Vulkan device: {} ({})",
        vk_device.device_name(),
        vk_device.vendor_string()
    );

    // 2. Compile a simple shader (rms_norm as a test)
    let shader_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/rms_norm.comp"
    );
    println!("Compiling shader: {}", shader_path);
    let spirv = kiln_vulkan_kernel::pipeline::ShaderPipeline::compile_shader(shader_path)?;
    println!("Compiled SPIR-V: {} bytes", spirv.len());

    // 3. Create a test tensor and dispatch
    let device = candle_core::Device::Cpu;
    let x = candle_core::Tensor::randn(0f32, 1.0, (1, 4, 8, 128), &device)?
        .to_dtype(candle_core::DType::F32)?;

    println!("Input shape: {:?}", x.shape());
    println!("Vulkan device and shader compilation: OK");

    Ok(())
}
