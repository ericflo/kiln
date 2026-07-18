use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ROCM");

    if std::env::var_os("CARGO_FEATURE_ROCM").is_some() {
        build_rocm();
    }

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

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
    let cuda_archs = env::var("KILN_CUDA_ARCHS").unwrap_or_else(|_| "80;89;90".to_string());

    let mut build = cc::Build::new();

    // Use nvcc as the compiler
    build.cuda(true);
    build.cpp(true);
    // Never emit device debug info (`-G`) for these kernels, even in a debug
    // (`cargo build`) profile. cc-rs adds `-G` when the cargo profile is debug,
    // but `-G` on the cutlass FlashAttention template instantiations balloons
    // nvcc's peak memory to tens of GB per source file — enough to get a single
    // compile OOM-SIGKILLed (exit 137) inside a typical container cgroup, so a
    // plain `cargo build --features cuda` of kiln-model fails on flash-attn. The
    // kernels are always `-O3` (set below) and are never gdb'd, so dropping the
    // debug flags is free. (`debug(false)` also drops host `-g`.)
    build.debug(false);
    configure_nvcc_from_cuda_root(&cuda_root);

    // Include paths
    build.include(&flash_attn_dir); // For "src/flash.h" etc.
    build.include(&kernel_src_dir); // For headers included without "src/" prefix
    build.include(&cutlass_include); // For <cutlass/...> and <cute/...>
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
    build.flag("-diag-suppress=177"); // variable declared but never referenced
    build.flag("-diag-suppress=174"); // expression has no effect
    // Host compiler emits these on vendored CUTLASS headers; nothing we can fix upstream.
    // GCC/clang-only — `-Wno-unused-parameter` becomes `/Wno-unused-parameter` to cl.exe
    // on Windows, which rejects it with D8021 ("invalid numeric argument"). cl.exe doesn't
    // emit the GCC-style "unused parameter" warning anyway, so there's nothing to silence.
    let target = env::var("TARGET").unwrap_or_default();
    if !target.ends_with("-msvc") {
        build.flag("-Xcompiler").flag("-Wno-unused-parameter");
    }

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
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_root.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=cudart");

    // Re-run if sources change
    println!("cargo:rerun-if-changed=csrc/");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=KILN_CUDA_ARCHS");
}

fn build_rocm() {
    let rocm_root = match find_rocm_root() {
        Some(p) => p,
        None => {
            println!(
                "cargo:warning=ROCm not found, kiln-flash-attn ROCm native kernels will not be \
                 compiled. Set ROCM_PATH or HIP_PATH, or install ROCm to /opt/rocm."
            );
            return;
        }
    };

    let hipcc = env::var("HIPCC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| rocm_root.join("bin").join("hipcc"));
    if !hipcc.exists() {
        println!(
            "cargo:warning=hipcc not found at {}; skipping ROCm native kernel build.",
            hipcc.display()
        );
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc_dir = manifest_dir.join("csrc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let archs =
        env::var("KILN_ROCM_ARCHS").unwrap_or_else(|_| "gfx942;gfx90a;gfx1100;gfx1151".to_string());

    let src = csrc_dir.join("rocm_flash_api.cpp");
    let obj = out_dir.join("rocm_flash_api.o");
    let mut cmd = std::process::Command::new(&hipcc);
    cmd.arg("-O3")
        .arg("-std=c++17")
        .arg("-fPIC")
        .arg("-c")
        .arg("-I")
        .arg(&csrc_dir)
        .arg("-I")
        .arg(rocm_root.join("include"));
    for arch in archs.split(';') {
        let arch = arch.trim();
        if !arch.is_empty() {
            cmd.arg(format!("--offload-arch={arch}"));
        }
    }
    if env::var("KILN_ROCM_WAVE64").is_ok() {
        cmd.arg("-mwavefrontsize64");
    }
    cmd.arg(&src).arg("-o").arg(&obj);
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn hipcc ({}): {e}", hipcc.display()));
    if !status.success() {
        panic!("hipcc failed to compile {}", src.display());
    }

    let lib_path = out_dir.join("libkiln_flash_attn_rocm.a");
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let _ = std::fs::remove_file(&lib_path);
    let status = std::process::Command::new(&ar)
        .arg("crus")
        .arg(&lib_path)
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn ar ({ar}): {e}"));
    if !status.success() {
        panic!("ar failed to create {}", lib_path.display());
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=kiln_flash_attn_rocm");
    println!(
        "cargo:rustc-link-search=native={}",
        rocm_root.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed=csrc/rocm_flash_api.cpp");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=KILN_ROCM_ARCHS");
    println!("cargo:rerun-if-env-changed=KILN_ROCM_WAVE64");
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
    if let Ok(output) = std::process::Command::new("which").arg("hipcc").output() {
        if output.status.success() {
            let hipcc_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let p = PathBuf::from(hipcc_path);
            if let Some(bin_dir) = p.parent() {
                if let Some(rocm_dir) = bin_dir.parent() {
                    return Some(rocm_dir.to_path_buf());
                }
            }
        }
    }
    None
}
