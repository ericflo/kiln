use std::env;
use std::path::PathBuf;

fn main() {
    // Only build when CUDA is available
    let cuda_root = find_cuda_root();
    let cuda_root = match cuda_root {
        Some(p) => p,
        None => {
            println!("cargo:warning=CUDA not found, kiln-flash-attn will not compile CUDA kernels");
            println!("cargo:warning=Set CUDA_ROOT or CUDA_HOME, or install CUDA toolkit");
            return;
        }
    };

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc_dir = manifest_dir.join("csrc");
    let flash_attn_dir = csrc_dir.join("flash_attn");
    let kernel_src_dir = flash_attn_dir.join("src");
    let cutlass_include = csrc_dir.join("cutlass");

    // Detect GPU architecture
    let cuda_archs = env::var("KILN_CUDA_ARCHS")
        .unwrap_or_else(|_| "80;89;90".to_string());

    let mut build = cc::Build::new();

    // Use nvcc as the compiler
    build.cuda(true);
    build.cpp(true);

    // Include paths
    build.include(&flash_attn_dir);        // For "src/flash.h" etc.
    build.include(&kernel_src_dir);         // For headers included without "src/" prefix
    build.include(&cutlass_include);        // For <cutlass/...> and <cute/...>
    build.include(cuda_root.join("include")); // CUDA headers

    // CUDA compilation flags
    build.flag("-std=c++17");
    build.flag("-O3");
    build.flag("--use_fast_math");
    build.flag("--expt-relaxed-constexpr");
    build.flag("--expt-extended-lambda");
    build.flag("-Xcompiler").flag("-fPIC");
    build.flag("-DFLASH_NAMESPACE=kiln_flash");
    build.flag("-D_USE_MATH_DEFINES");

    // Suppress noisy warnings from CUTLASS templates
    build.flag("-diag-suppress=177");  // variable declared but never referenced
    build.flag("-diag-suppress=174");  // expression has no effect

    // Architecture flags — only sm80+ (flash-attn requirement)
    for arch in cuda_archs.split(';') {
        let arch = arch.trim();
        if !arch.is_empty() {
            build.flag(&format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
        }
    }

    // Source files to compile:
    // 1. Our C-ABI wrapper
    build.file(flash_attn_dir.join("flash_api_c.cu"));
    // 2. The template instantiation files (bf16, causal, hdim128 + hdim256)
    build.file(kernel_src_dir.join("flash_fwd_hdim128_bf16_causal_sm80.cu"));
    build.file(kernel_src_dir.join("flash_bwd_hdim128_bf16_causal_sm80.cu"));
    build.file(kernel_src_dir.join("flash_fwd_split_hdim128_bf16_causal_sm80.cu"));
    build.file(kernel_src_dir.join("flash_fwd_hdim256_bf16_causal_sm80.cu"));
    build.file(kernel_src_dir.join("flash_bwd_hdim256_bf16_causal_sm80.cu"));
    build.file(kernel_src_dir.join("flash_fwd_split_hdim256_bf16_causal_sm80.cu"));

    // Compile
    build.compile("kiln_flash_attn");

    // Link CUDA runtime
    println!("cargo:rustc-link-search=native={}", cuda_root.join("lib64").display());
    println!("cargo:rustc-link-lib=cudart");

    // Re-run if sources change
    println!("cargo:rerun-if-changed=csrc/");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=KILN_CUDA_ARCHS");
}

fn find_cuda_root() -> Option<PathBuf> {
    // Check env vars first
    for var in &["CUDA_ROOT", "CUDA_HOME", "CUDA_PATH"] {
        if let Ok(val) = env::var(var) {
            let p = PathBuf::from(val);
            if p.join("include").join("cuda.h").exists() {
                return Some(p);
            }
        }
    }
    // Check common locations
    for path in &[
        "/usr/local/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.6",
        "/opt/cuda",
    ] {
        let p = PathBuf::from(path);
        if p.join("include").join("cuda.h").exists() {
            return Some(p);
        }
    }
    // Try nvcc in PATH
    if let Ok(output) = std::process::Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let nvcc_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            // nvcc is typically at <cuda_root>/bin/nvcc
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
