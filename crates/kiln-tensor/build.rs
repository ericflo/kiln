// Build script for kiln-tensor.
//
// Two responsibilities, each gated on a Cargo feature:
//
// 1. `--features cuda` (existing): compiles the CUDA-side `contiguous`
//    kernel family (`csrc/*.cu`) into a static library that the Rust
//    side links against. Outside the cuda feature this part is a no-op.
//
// 2. `--features vulkan` (new, #1082): compiles GLSL compute shaders
//    under `shaders/*.comp` to SPIR-V using `glslc` (or
//    `glslangValidator` as a fallback), then embeds the resulting
//    `.spv` bytes into the crate via an auto-generated `kt_spirv.rs`
//    in `OUT_DIR`. Mirrors the pattern used by
//    `crates/kiln-vulkan-kernel/build.rs`. Outside the vulkan feature
//    this part is a no-op.
//
// Both paths are kept in this single build script so cargo only has to
// re-run one build for the crate. The two paths are otherwise
// independent.

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Vulkan compute shaders to compile under `shaders/*.comp`.
///
/// Format: `(base_name, const_ident)`. `base_name.comp` must exist
/// under `crates/kiln-tensor/shaders/`. The generated `kt_spirv.rs`
/// will export `pub const <const_ident>: &[u8] = include_bytes!(...)`.
///
/// New shaders go here; no other build-script edit needed.
const VK_SHADERS: &[(&str, &str)] = &[
    // Trivial f32 copy kernel — proof-of-concept that the shader
    // build pipeline works end-to-end inside kiln-tensor.
    ("kt_identity_f32", "SPIR_V_KT_IDENTITY_F32"),
];

fn main() {
    // --- CUDA: existing path (unchanged) ---
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    if cuda_enabled {
        build_cuda();
    }

    // --- VULKAN: new shader compile path (#1082) ---
    let vulkan_enabled = env::var("CARGO_FEATURE_VULKAN").is_ok();
    // The Rust side's `vk_shaders` module always tries to include the
    // generated `kt_spirv.rs`. When the vulkan feature is off, write an
    // empty stub module so the include resolves cleanly (the module
    // itself is also `cfg(feature = "vulkan")`-gated, so the file is
    // only included into the crate when the feature is on — but we
    // still need the file to exist on every build to avoid an
    // include_str/include_bytes failure if someone toggles features
    // without `cargo clean`).
    build_vulkan_shaders(vulkan_enabled);
}

// ---------------------------------------------------------------------
// CUDA path (unchanged from prior build.rs)
// ---------------------------------------------------------------------

fn build_cuda() {
    let cuda_root = match find_cuda_root() {
        Some(p) => p,
        None => {
            println!(
                "cargo:warning=CUDA not found, kiln-tensor's CUDA-side contiguous kernel \
                 will not be compiled. Set CUDA_ROOT or CUDA_HOME, or install CUDA toolkit."
            );
            return;
        }
    };

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc_dir = manifest_dir.join("csrc");
    if !csrc_dir.exists() {
        return;
    }

    let cuda_archs = env::var("KILN_CUDA_ARCHS").unwrap_or_else(|_| "80;86;89;90".to_string());

    configure_nvcc_from_cuda_root(&cuda_root);

    let mut build = cc::Build::new();
    build.cuda(true);
    build.cpp(true);

    build.include(&csrc_dir);
    build.include(cuda_root.join("include"));

    build.flag("-std=c++17");
    build.flag("-O3");
    build.flag("--use_fast_math");
    build.flag("--expt-relaxed-constexpr");
    build.flag("-Xcompiler").flag("-fPIC");

    for arch in cuda_archs.split(';') {
        let arch = arch.trim();
        if !arch.is_empty() {
            build.flag(&format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
        }
    }

    build.file(csrc_dir.join("contiguous.cu"));
    build.file(csrc_dir.join("index_select.cu"));
    build.file(csrc_dir.join("elementwise.cu"));
    build.file(csrc_dir.join("activation.cu"));
    build.file(csrc_dir.join("cast.cu"));
    build.file(csrc_dir.join("softmax.cu"));
    build.file(csrc_dir.join("reduce_last_axis.cu"));
    build.file(csrc_dir.join("reduce_arbitrary_axis.cu"));
    build.file(csrc_dir.join("argmax_last_axis.cu"));
    build.file(csrc_dir.join("masked_fill.cu"));
    build.file(csrc_dir.join("scatter_add.cu"));
    build.file(csrc_dir.join("cross_entropy.cu"));
    build.file(csrc_dir.join("concat.cu"));
    build.file(csrc_dir.join("rope.cu"));
    build.file(csrc_dir.join("dropout.cu"));
    build.file(csrc_dir.join("rmsnorm.cu"));
    build.file(csrc_dir.join("layernorm.cu"));
    build.file(csrc_dir.join("scalar_op.cu"));
    build.file(csrc_dir.join("clamp_pow.cu"));
    build.file(csrc_dir.join("compare.cu"));
    build.file(csrc_dir.join("where_select.cu"));
    build.file(csrc_dir.join("diag.cu"));
    build.file(csrc_dir.join("binary_minmax.cu"));
    build.file(csrc_dir.join("lerp.cu"));
    build.compile("kiln_tensor_cuda_ops");

    println!(
        "cargo:rustc-link-search=native={}",
        cuda_root.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=cudart");

    println!("cargo:rerun-if-changed=csrc/");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=KILN_CUDA_ARCHS");
}

fn configure_nvcc_from_cuda_root(cuda_root: &PathBuf) {
    if env::var_os("NVCC").is_some() {
        return;
    }
    let nvcc = cuda_root.join("bin").join("nvcc");
    if nvcc.exists() {
        unsafe {
            env::set_var("NVCC", nvcc);
        }
    }
}

fn find_cuda_root() -> Option<PathBuf> {
    for var in &["CUDA_ROOT", "CUDA_HOME", "CUDA_PATH"] {
        if let Ok(val) = env::var(var) {
            let p = PathBuf::from(val);
            if p.join("include").join("cuda.h").exists() {
                return Some(p);
            }
        }
    }
    for path in &[
        "/usr/local/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12.8",
        "/opt/cuda",
    ] {
        let p = PathBuf::from(path);
        if p.join("include").join("cuda.h").exists() {
            return Some(p);
        }
    }
    if let Ok(output) = std::process::Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let nvcc_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let p = PathBuf::from(nvcc_path);
            if let Some(bin_dir) = p.parent() {
                if let Some(cuda_dir) = bin_dir.parent() {
                    if cuda_dir.join("include").join("cuda.h").exists() {
                        return Some(cuda_dir.to_path_buf());
                    }
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------
// Vulkan shader compile path (#1082)
// ---------------------------------------------------------------------

fn build_vulkan_shaders(vulkan_enabled: bool) {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders_dir = manifest_dir.join("shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::create_dir_all(&out_dir).expect("OUT_DIR must be creatable");

    // Always emit `kt_spirv.rs` so the include in `src/vk_shaders.rs`
    // never fails. When the vulkan feature is off OR there are no
    // shaders to compile, we emit an empty `spirv_modules` mod stub.
    let mut emit_shaders: Vec<(&str, &str, Vec<u8>)> = Vec::new();

    if vulkan_enabled && shaders_dir.exists() {
        for (name, var_name) in VK_SHADERS {
            let glsl_path = shaders_dir.join(format!("{}.comp", name));
            let spv_path = out_dir.join(format!("{}.spv", name));

            if !glsl_path.exists() {
                eprintln!(
                    "cargo:warning=kiln-tensor: GLSL shader not found: {}",
                    glsl_path.display()
                );
                continue;
            }

            match compile_shader_command(&glsl_path, &spv_path) {
                Ok(output) if output.status.success() => {
                    let bytes = fs::read(&spv_path).unwrap_or_default();
                    emit_shaders.push((*name, *var_name, bytes));
                }
                Ok(output) => {
                    eprintln!(
                        "cargo:warning=kiln-tensor: glslc failed for {}: {}",
                        name,
                        String::from_utf8_lossy(&output.stderr)
                    );
                    // Emit an empty placeholder so dependent Rust code
                    // still compiles; runtime use will see len==0 and
                    // can error cleanly.
                    emit_shaders.push((*name, *var_name, Vec::new()));
                }
                Err(e) => {
                    eprintln!(
                        "cargo:warning=kiln-tensor: glslc/glslangValidator not found ({}) \
                         — Vulkan shaders will not be embedded for this build",
                        e
                    );
                    emit_shaders.push((*name, *var_name, Vec::new()));
                }
            }
        }
    } else if vulkan_enabled && !shaders_dir.exists() {
        eprintln!(
            "cargo:warning=kiln-tensor: vulkan feature enabled but {} does not exist \
             — no shaders to compile",
            shaders_dir.display()
        );
    }

    // Generate `kt_spirv.rs` regardless. When `vulkan_enabled` is
    // false we still generate the module with empty entries so the
    // crate compiles on hosts without glslc.
    let mut out = String::new();
    out.push_str("// Auto-generated by kiln-tensor/build.rs — do not edit.\n");
    out.push_str("// Vulkan SPIR-V shader modules embedded at build time.\n\n");
    out.push_str("#[rustfmt::skip]\n");
    out.push_str("pub mod spirv_modules {\n");

    if emit_shaders.is_empty() {
        out.push_str("    // (no shaders compiled in this configuration)\n");
    } else {
        for (name, var_name, bytes) in &emit_shaders {
            let spv_path = out_dir.join(format!("{}.spv", name));
            if !bytes.is_empty() && spv_path.exists() {
                let len = bytes.len();
                out.push_str(&format!(
                    "    /// Embedded SPIR-V for {}.comp ({len} bytes).\n",
                    name
                ));
                out.push_str(&format!(
                    "    pub const {}: &[u8] = include_bytes!(\"{}\");\n\n",
                    var_name,
                    spv_path.display()
                ));
            } else {
                out.push_str(&format!(
                    "    /// Shader {} not compiled at build time (glslc missing or failed).\n",
                    name
                ));
                out.push_str(&format!("    pub const {}: &[u8] = &[];\n\n", var_name));
            }
        }
    }
    out.push_str("}\n");

    let generated_path = out_dir.join("kt_spirv.rs");
    let mut file = fs::File::create(&generated_path).expect("write kt_spirv.rs");
    file.write_all(out.as_bytes()).expect("write kt_spirv.rs");

    println!("cargo:rerun-if-changed=shaders/");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PATH");
}

/// Invoke `glslc` (or `glslangValidator` as a fallback) on
/// `glsl_path`, writing SPIR-V output to `spv_path`. Matches the
/// invocation used by `crates/kiln-vulkan-kernel/build.rs` so the
/// same kiln Vulkan device (instance v1.2, target-env vulkan1.1)
/// accepts the resulting bytecode.
fn compile_shader_command(
    glsl_path: &Path,
    spv_path: &Path,
) -> std::io::Result<std::process::Output> {
    let glslc = std::process::Command::new("glslc")
        .arg(glsl_path)
        .arg("-o")
        .arg(spv_path)
        .arg("--target-env=vulkan1.1")
        .arg("-DFLOAT_TYPE=float")
        .arg("-DUSE_BFLOAT16=1")
        .arg("-DUSE_SUBGROUP_ADD=1")
        .arg("-DUSE_SUBGROUP_CLUSTERED=1")
        .output();
    if glslc.is_ok() {
        return glslc;
    }

    std::process::Command::new("glslangValidator")
        .arg("-V")
        .arg(glsl_path)
        .arg("-o")
        .arg(spv_path)
        .arg("--target-env")
        .arg("vulkan1.1")
        .arg("-DFLOAT_TYPE=float")
        .arg("-DUSE_BFLOAT16=1")
        .arg("-DUSE_SUBGROUP_ADD=1")
        .arg("-DUSE_SUBGROUP_CLUSTERED=1")
        .output()
}
