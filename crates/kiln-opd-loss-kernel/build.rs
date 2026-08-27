//! Build script for the fused OPD top-K reverse-KL CUDA kernel.
//!
//! Mirrors `kiln-rmsnorm-kernel/build.rs`. Gracefully no-ops on hosts
//! without CUDA so `cargo check` / `cargo build` work everywhere; the
//! CUDA kernel only compiles when the toolchain is present AND the
//! crate's `cuda` feature is on (the latter gates the FFI extern
//! declaration in `src/phase_b.rs`).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // --- ROCm: hipcc kernel compile path (Phase R.7) ---
    // Gated on the `rocm` feature, mirroring `kiln-tensor/build.rs`.
    if env::var("CARGO_FEATURE_ROCM").is_ok() {
        build_rocm();
    }

    // --- CUDA: existing path (unchanged) ---
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda();
    }
}

fn build_cuda() {
    let cuda_root = match find_cuda_root() {
        Some(p) => p,
        None => {
            println!(
                "cargo:warning=CUDA not found, kiln-opd-loss-kernel will not compile CUDA kernels"
            );
            println!("cargo:warning=Set CUDA_ROOT or CUDA_HOME, or install CUDA toolkit");
            return;
        }
    };

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc_dir = manifest_dir.join("csrc");

    // (Phase R.7) The kernel now `#include "kt_gpu_compat.cuh"` (the shared
    // wave-size shim), which lives in `kiln-tensor/csrc`. Add it to the nvcc
    // include path so the CUDA build resolves it too — the helper compiles to
    // identical CUDA code (KILN_WARP == warpSize == 32 on NVIDIA), so CUDA
    // behavior is unchanged.
    let kt_csrc = manifest_dir
        .parent()
        .expect("crate dir has a parent (crates/)")
        .join("kiln-tensor")
        .join("csrc");

    let cuda_archs = env::var("KILN_CUDA_ARCHS").unwrap_or_else(|_| "80;86;89;90".to_string());

    let mut build = cc::Build::new();
    build.cuda(true);
    build.cpp(true);
    configure_nvcc_from_cuda_root(&cuda_root);

    build.include(&csrc_dir);
    build.include(&kt_csrc);
    build.include(cuda_root.join("include"));

    build.flag("-std=c++17");
    build.flag("-O3");
    build.flag("--use_fast_math");
    build.flag("--expt-relaxed-constexpr");
    build.flag("--expt-extended-lambda");
    build.flag("-Xcompiler").flag("-fPIC");
    // Same nvcc diag suppressions rmsnorm uses — silence warnings about
    // mixed-type comparisons and casts in templated code.
    build.flag("-diag-suppress=177");
    build.flag("-diag-suppress=174");

    for arch in cuda_archs.split(';') {
        let arch = arch.trim();
        if !arch.is_empty() {
            build.flag(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
        }
    }

    build.file(csrc_dir.join("opd_topk_kl.cu"));

    build.compile("kiln_opd_loss_kernel");

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

fn configure_nvcc_from_cuda_root(cuda_root: &Path) {
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
    None
}

// ---------------------------------------------------------------------
// ROCm / HIP kernel compile path (Phase R.7)
// ---------------------------------------------------------------------

/// hipcc-compile the fused OPD top-K reverse-KL backward kernel
/// (`csrc/opd_topk_kl.cu`) into a static lib that the Rust side links
/// against. `cc::Build::cuda(true)` is hardwired to nvcc, so we drive
/// `hipcc` directly — exactly the pattern in `kiln-tensor/build.rs`.
/// No-op (with a `cargo:warning`) when ROCm is absent so `cargo check
/// --features rocm` stays green on a toolchain-less host.
///
/// The kernel `#include`s `<cuda_bf16.h>` / `<cuda_runtime.h>`, which the
/// shared CUDA->HIP compat shim under `kiln-tensor/csrc/hip_compat/`
/// resolves to HIP headers; the wave-size shim `kt_gpu_compat.cuh` lives
/// in `kiln-tensor/csrc/`. We add `-I` for both of those plus this
/// crate's own `csrc`.
fn build_rocm() {
    let rocm_root = match find_rocm_root() {
        Some(p) => p,
        None => {
            println!(
                "cargo:warning=ROCm not found, kiln-opd-loss-kernel's ROCm kernel will not be \
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
            "cargo:warning=hipcc not found at {}; skipping ROCm kernel build.",
            hipcc.display()
        );
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc_dir = manifest_dir.join("csrc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Shared compat + wave-size shim headers live in `kiln-tensor/csrc`
    // (and its `hip_compat/` subdir). Compute the path relative to this
    // crate's manifest so it works regardless of the absolute checkout.
    let kt_csrc = manifest_dir
        .parent()
        .expect("crate dir has a parent (crates/)")
        .join("kiln-tensor")
        .join("csrc");
    let kt_compat = kt_csrc.join("hip_compat");

    // Local builds target the installed GPU. Release and cross builds can
    // request a semicolon-separated fat binary through KILN_ROCM_ARCHS.
    let archs = env::var("KILN_ROCM_ARCHS").unwrap_or_else(|_| "native".to_string());

    // The fused OPD top-K reverse-KL backward kernel. Its two cross-lane
    // reductions were converted to wave-agnostic `kiln_block_reduce_*`
    // (Phase R.7) so it is correct on AMD wave32 + wave64 + NVIDIA.
    const ROCM_KERNELS: &[&str] = &["opd_topk_kl.cu"];

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
        // KILN_ROCM_WAVE64: force 64-lane wavefronts so the parity oracle can
        // validate the wave-size-fixed reductions under real wave64 execution
        // even on an RDNA dev box.
        if env::var("KILN_ROCM_WAVE64").is_ok() {
            cmd.arg("-mwavefrontsize64");
        }
        cmd.arg("-I").arg(&kt_compat).arg("-I").arg(&kt_csrc);
        cmd.arg("-I").arg(&csrc_dir);
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
    let lib = out_dir.join("libkiln_opd_loss_kernel_rocm_ops.a");
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
    println!("cargo:rustc-link-lib=static=kiln_opd_loss_kernel_rocm_ops");
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
    if let Ok(output) = Command::new("which").arg("hipcc").output()
        && output.status.success()
    {
        let hipcc_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let p = PathBuf::from(hipcc_path);
        if let Some(bin_dir) = p.parent()
            && let Some(root) = bin_dir.parent()
        {
            return Some(root.to_path_buf());
        }
    }
    None
}
