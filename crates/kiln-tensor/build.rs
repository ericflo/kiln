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
use std::process::Command;

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

    // --- ROCm: hipcc kernel compile path (Phase R.2) ---
    let rocm_enabled = env::var("CARGO_FEATURE_ROCM").is_ok();
    if rocm_enabled {
        build_rocm();
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
    build.file(csrc_dir.join("topk_last_axis.cu"));
    build.file(csrc_dir.join("masked_fill.cu"));
    build.file(csrc_dir.join("scatter_add.cu"));
    build.file(csrc_dir.join("cross_entropy.cu"));
    build.file(csrc_dir.join("flce_grad.cu"));
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
    build.file(csrc_dir.join("scan_axis.cu"));
    build.file(csrc_dir.join("lerp.cu"));
    build.file(csrc_dir.join("fp8.cu"));
    build.file(csrc_dir.join("is_finite_reduce.cu"));
    build.file(csrc_dir.join("paged_decode_meta.cu"));
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
// ROCm / HIP kernel compile path (Phase R.2)
// ---------------------------------------------------------------------

/// hipcc-compile the ROCm-side `csrc/*.cu` kernels into a static lib that the
/// Rust side links against. `cc::Build::cuda(true)` is hardwired to nvcc, so we
/// drive `hipcc` directly. No-op (with a `cargo:warning`) when ROCm is absent,
/// so `cargo check --features rocm` stays green on a toolchain-less host.
///
/// The CUDA->HIP compat shim (`csrc/hip_compat/`) lets the `.cu` sources
/// compile byte-unchanged: their `#include <cuda_runtime.h>` resolves to the
/// shim instead of NVIDIA's header.
fn build_rocm() {
    let rocm_root = match find_rocm_root() {
        Some(p) => p,
        None => {
            println!(
                "cargo:warning=ROCm not found, kiln-tensor's ROCm kernels will not be compiled. \
                 Set ROCM_PATH or HIP_PATH, or install ROCm to /opt/rocm."
            );
            return;
        }
    };

    let hipcc = env::var("HIPCC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| rocm_root.join("bin").join("hipcc"));
    if !hipcc.exists() {
        println!(
            "cargo:warning=hipcc not found at {}; skipping ROCm kernel build.",
            hipcc.display()
        );
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc_dir = manifest_dir.join("csrc");
    let compat_dir = csrc_dir.join("hip_compat");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // gfx targets: CDNA (gfx90a/gfx942) + RDNA3 (gfx1100) per the v1 matrix,
    // plus gfx1151 (Strix Halo) so on-hardware parity testing runs on the dev
    // box. Override with KILN_ROCM_ARCHS.
    let archs =
        env::var("KILN_ROCM_ARCHS").unwrap_or_else(|_| "gfx942;gfx90a;gfx1100;gfx1151".to_string());

    // Hipify-clean kernels (R.2) + reduction kernels that have taken the
    // Phase R.5 wave-size fix (routed through `kt_gpu_compat.cuh`'s
    // `KILN_FULL_MASK` — HIP 7.x static_asserts a 64-bit shuffle mask). The
    // remaining reduction family joins as each kernel is wave-audited +
    // parity-swept on real wave64 hardware.
    const ROCM_KERNELS: &[&str] = &[
        // R.2 baseline
        "contiguous.cu",
        "elementwise.cu",
        // R.5 — reductions (wave-size fixed via kiln_block_reduce_*)
        "softmax.cu",
        "reduce_last_axis.cu",
        "reduce_arbitrary_axis.cu",
        "argmax_last_axis.cu",
        "cross_entropy.cu",
        "rmsnorm.cu",
        "layernorm.cu",
        "is_finite_reduce.cu",
        // R.5 — hipify-clean elementwise / map kernels
        "cast.cu",
        "activation.cu",
        "index_select.cu",
        // R.9 — on-device scatter-copy (inverse of index_select); the device-slot
        // paged-KV write that lets the K/V store record into a HIP graph.
        "index_copy.cu",
        "scalar_op.cu",
        "rope.cu",
        "masked_fill.cu",
        "where_select.cu",
        "compare.cu",
        "concat.cu",
        "binary_minmax.cu",
        "clamp_pow.cu",
        "lerp.cu",
        "diag.cu",
        // R.5b — deferred GDN/sampling/training hot-path kernels.
        // scan_axis: cumsum/cumprod (GDN cumsum hot path); smem-only scan, no
        //   cross-lane reduction → wave32/64-correct as-is.
        // scatter_add: dim0 atomic scatter (embedding-bwd); F32 native atomicAdd,
        //   BF16 via CAS-on-dword (HIP has no native bf16 atomicAdd).
        // dropout: counter-based splitmix64 RNG (no curand/hiprand dependency).
        "scan_axis.cu",
        "scatter_add.cu",
        "dropout.cu",
        // R.5b — on-device sampling top-k (keeps the full [V] logits row
        // resident, returns only k (value,index) pairs — vs the host-sort
        // fallback's full-[V] D2H every sampled token). The per-pass argmax
        // reduction uses 32-lane subgroups explicitly (warp_id=tid/32,
        // shfl offset<=16), so it is wave32/64-correct as-is.
        "topk_last_axis.cu",
        // R.5b — FP8 (E4M3FN) KV-cache quantize/dequantize. Pure elementwise
        // bit math (portable f32<->e4m3 encode/decode, no hardware-fp8
        // intrinsics, no cross-lane reduction) → hipcc-clean + wave-agnostic.
        "fp8.cu",
        // R.9 prereq — on-device paged-decode metadata (device block_table ->
        // gather index, device seqused_k -> tail mask); kills the per-attn-layer
        // D2H/H2D round-trip and unblocks HIP graph capture. Pure index math,
        // no cross-lane reduction → wave32/64-correct.
        "paged_decode_meta.cu",
        // Experimental decode-only W8A16 GEMV. One block per output row with a
        // wave-size-agnostic shared-memory reduction; used behind an opt-in
        // model-side flag for bandwidth-bound single-token projections.
        "w8_gemv.cu",
    ];

    let mut objects = Vec::new();
    for kernel in ROCM_KERNELS {
        let src = csrc_dir.join(kernel);
        let obj = out_dir.join(format!("{kernel}.o"));
        let mut cmd = Command::new(&hipcc);
        cmd.arg("-O3").arg("-std=c++17").arg("-fPIC").arg("-c");
        for arch in archs.split(';') {
            let a = arch.trim();
            if !a.is_empty() {
                cmd.arg(format!("--offload-arch={a}"));
            }
        }
        // KILN_ROCM_WAVE64: force 64-lane wavefronts. CDNA (gfx9xx) is always
        // wave64; RDNA (gfx10/11) defaults to wave32 but can run wave64. The
        // parity oracle sets this to validate the wave-size-fixed reductions
        // under real wave64 execution even on an RDNA dev box, de-risking CDNA
        // deployment without CDNA hardware.
        if env::var("KILN_ROCM_WAVE64").is_ok() {
            cmd.arg("-mwavefrontsize64");
        }
        cmd.arg("-I").arg(&compat_dir).arg("-I").arg(&csrc_dir);
        cmd.arg(&src).arg("-o").arg(&obj);
        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("failed to spawn hipcc ({}): {e}", hipcc.display()));
        if !status.success() {
            panic!("hipcc failed to compile {kernel}");
        }
        objects.push(obj);
    }

    // Archive the device objects into a static lib.
    let lib = out_dir.join("libkiln_tensor_rocm_ops.a");
    let _ = fs::remove_file(&lib);
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib);
    for o in &objects {
        ar.arg(o);
    }
    let status = ar.status().expect("failed to spawn ar");
    if !status.success() {
        panic!("ar failed to archive ROCm kernels into {}", lib.display());
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=kiln_tensor_rocm_ops");
    println!(
        "cargo:rustc-link-search=native={}",
        rocm_root.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    // hipcc device objects pull in the C++ runtime at final link.
    println!("cargo:rustc-link-lib=dylib=stdc++");

    println!("cargo:rerun-if-changed=csrc/");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=KILN_ROCM_ARCHS");
    println!("cargo:rerun-if-env-changed=KILN_ROCM_WAVE64");
}

/// Locate a ROCm install root containing `bin/hipcc`. Honours `ROCM_PATH` /
/// `HIP_PATH`, then `/opt/rocm`, then `which hipcc`.
fn find_rocm_root() -> Option<PathBuf> {
    for var in &["ROCM_PATH", "HIP_PATH"] {
        if let Ok(val) = env::var(var) {
            let p = PathBuf::from(val);
            if p.join("bin").join("hipcc").exists() {
                return Some(p);
            }
        }
    }
    let default = PathBuf::from("/opt/rocm");
    if default.join("bin").join("hipcc").exists() {
        return Some(default);
    }
    if let Ok(output) = Command::new("which").arg("hipcc").output() {
        if output.status.success() {
            let hipcc_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let p = PathBuf::from(hipcc_path);
            if let Some(bin_dir) = p.parent() {
                if let Some(root) = bin_dir.parent() {
                    return Some(root.to_path_buf());
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
