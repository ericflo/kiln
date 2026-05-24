// Build script for kiln-tensor.
//
// Under `--features cuda`, compiles the CUDA-side `contiguous` kernel
// (`csrc/contiguous.cu`) into a static library that the Rust side
// links against. Outside the cuda feature, this is a no-op.

use std::env;
use std::path::PathBuf;

fn main() {
    // Only compile CUDA code when the cuda feature is enabled.
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    if !cuda_enabled {
        return;
    }

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
