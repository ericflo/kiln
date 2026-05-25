//! Vulkan SPIR-V shader bytes embedded by `build.rs` from
//! `crates/kiln-tensor/shaders/*.comp`.
//!
//! This module is the kiln-tensor counterpart to
//! `kiln_vulkan_kernel::shaders` — it hosts compute shaders that
//! belong with the tensor substrate itself (e.g. layout-aware
//! concat / chunk / split implementations) rather than with the
//! model-specific kernels in `kiln-vulkan-kernel`.
//!
//! # How to add a new shader
//!
//! 1. Drop the `.comp` source into `crates/kiln-tensor/shaders/`.
//! 2. Register it in `VK_SHADERS` in `crates/kiln-tensor/build.rs`
//!    with a base name and a `SCREAMING_SNAKE_CASE` constant ident.
//! 3. The auto-generated `kt_spirv.rs` will expose
//!    `kiln_tensor::vk_shaders::spirv_modules::<IDENT>: &[u8]` after
//!    the next `cargo build --features vulkan`.
//!
//! # Build-time vs runtime
//!
//! Mirrors `crates/kiln-vulkan-kernel/build.rs`:
//!
//! - `glslc` (or `glslangValidator`) is invoked at build time with
//!   `--target-env=vulkan1.1` so subgroup-op shaders compile cleanly
//!   on kiln's Vulkan 1.2 instance.
//! - If glslc is missing the build emits a `cargo:warning=` and the
//!   corresponding `&[u8]` constant is empty. Callers should check
//!   `is_empty()` before passing the bytes to a Vulkan pipeline
//!   create call and either fall back to a CPU path or surface a
//!   clear runtime error.
//!
//! # Proof-of-concept entry
//!
//! `SPIR_V_KT_IDENTITY_F32` is the bootstrap proof-of-concept: an
//! element-wise f32 copy that verifies the build pipeline works
//! end-to-end. Real ops (concat / chunk / split / etc.) replace it
//! in subsequent PRs.

// Include the auto-generated module from `OUT_DIR`. `build.rs`
// guarantees this file exists on every build, even when the vulkan
// feature is off or no shaders compiled successfully (in those cases
// the embedded `spirv_modules` module is empty / stubs to `&[]`).
include!(concat!(env!("OUT_DIR"), "/kt_spirv.rs"));

pub use spirv_modules::*;

/// Returns `true` if the build script successfully embedded SPIR-V
/// bytes for the trivial identity kernel. False on hosts without
/// `glslc` / `glslangValidator`. Useful in tests that want to skip
/// kernel-dispatch checks on shader-less CI environments.
#[allow(dead_code)]
pub fn identity_f32_available() -> bool {
    !SPIR_V_KT_IDENTITY_F32.is_empty()
}
