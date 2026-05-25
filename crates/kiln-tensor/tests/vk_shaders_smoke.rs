//! Smoke test for the kiln-tensor Vulkan shader build pipeline (#1082).
//!
//! Verifies that when the crate is built with `--features vulkan`, the
//! proof-of-concept identity shader is compiled to SPIR-V by `build.rs`
//! and embedded into the crate via `kiln_tensor::vk_shaders`.
//!
//! Outside the `vulkan` feature there's nothing to test (the module is
//! cfg-gated out), so the whole file compiles to an empty crate.
//!
//! This test does NOT instantiate a Vulkan device — it only validates
//! the build/embed pipeline. Actually dispatching the shader is
//! exercised by future ops once `VulkanStorage` grows H2D/D2H staging.

#![cfg(feature = "vulkan")]

use kiln_tensor::vk_shaders;

/// SPIR-V magic number, little-endian. Present at the start of every
/// well-formed SPIR-V module.
///
/// See SPIR-V spec §2.3 ("Physical Layout of a SPIR-V Module"):
/// `Magic Number = 0x07230203`.
const SPIRV_MAGIC: u32 = 0x0723_0203;

#[test]
fn identity_f32_shader_is_embedded_or_explicitly_unavailable() {
    let bytes = vk_shaders::SPIR_V_KT_IDENTITY_F32;

    // The build pipeline either embeds real SPIR-V bytes (when glslc
    // / glslangValidator is on PATH) or emits a clearly-empty
    // placeholder. Either outcome must agree with
    // `identity_f32_available()`.
    let available = vk_shaders::identity_f32_available();
    assert_eq!(available, !bytes.is_empty(),
        "vk_shaders::identity_f32_available() and SPIR_V_KT_IDENTITY_F32.is_empty() must agree");

    if !available {
        // glslc/glslangValidator was unavailable at build time.
        // The build emitted a `cargo:warning=` and the constant is
        // empty. Nothing further to validate.
        eprintln!(
            "kt_identity_f32: build-time SPIR-V not available (likely missing glslc); skipping magic-number check"
        );
        return;
    }

    // SPIR-V modules are u32-word streams. The byte length must be
    // a multiple of 4 and the first word must be the magic number.
    assert_eq!(
        bytes.len() % 4,
        0,
        "SPIR-V byte length {} is not a multiple of 4",
        bytes.len()
    );
    assert!(
        bytes.len() >= 20,
        "SPIR-V module suspiciously short ({} bytes); spec mandates at least a 5-word header",
        bytes.len()
    );

    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(
        magic, SPIRV_MAGIC,
        "First word of embedded kt_identity_f32.spv is 0x{:08x}, expected SPIR-V magic 0x{:08x}",
        magic, SPIRV_MAGIC
    );
}
