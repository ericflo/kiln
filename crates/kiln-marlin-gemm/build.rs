use std::env;
use std::path::PathBuf;

fn main() {
    let cuda_root = match find_cuda_root() {
        Some(p) => p,
        None => {
            println!(
                "cargo:warning=CUDA not found, kiln-marlin-gemm will not compile CUDA kernels"
            );
            println!("cargo:warning=Set CUDA_ROOT or CUDA_HOME, or install CUDA toolkit");
            return;
        }
    };

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc_dir = manifest_dir.join("csrc");

    // Marlin requires sm_80+ (cp.async, mma.m16n8k16, ldmatrix). We default to
    // the same arch list as the other kiln CUDA crates; users can still
    // override via KILN_CUDA_ARCHS if they need to drop a target.
    let cuda_archs =
        env::var("KILN_CUDA_ARCHS").unwrap_or_else(|_| "80;86;89;90".to_string());

    let mut build = cc::Build::new();
    build.cuda(true);
    build.cpp(true);

    build.include(&csrc_dir);
    build.include(cuda_root.join("include"));

    build.flag("-std=c++17");
    build.flag("-O3");
    build.flag("--use_fast_math");
    build.flag("--expt-relaxed-constexpr");
    build.flag("--expt-extended-lambda");
    build.flag("-Xcompiler").flag("-fPIC");
    // Suppress diag-suppress noise from the vendored kernel (unused vars in
    // the unused CALL_IF arms after templating).
    build.flag("-diag-suppress=177");
    build.flag("-diag-suppress=174");
    // The Marlin source uses `groupsize / thread_k_blocks` style expressions
    // inside CALL_IF arms that are unreachable for the per-column case
    // (group_blocks == -1). NVCC still type-checks the dead branches and warns
    // about division by zero / unreachable code; suppress those so the build
    // log stays focused on real issues.
    build.flag("-diag-suppress=39");
    build.flag("-diag-suppress=179");

    for arch in cuda_archs.split(';') {
        let arch = arch.trim();
        if !arch.is_empty() {
            build.flag(&format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
        }
    }

    build.file(csrc_dir.join("marlin_kernel.cu"));

    build.compile("kiln_marlin_gemm");

    println!(
        "cargo:rustc-link-search=native={}",
        cuda_root.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=cudart");

    println!("cargo:rerun-if-changed=csrc/");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=KILN_CUDA_ARCHS");
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
