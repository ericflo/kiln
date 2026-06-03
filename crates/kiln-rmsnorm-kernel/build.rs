// Build script for kiln-rmsnorm-kernel.
//
// Two GPU-kernel compile paths, each gated on a Cargo feature:
//
// 1. `--features cuda` (existing): nvcc-compiles `csrc/*.cu` into a static lib
//    via `cc::Build::cuda(true)`. No-op outside the cuda feature.
//
// 2. `--features rocm` (Phase R.7): hipcc-compiles the SAME `csrc/*.cu` into a
//    static lib. `cc::Build::cuda(true)` is hardwired to nvcc, so we drive
//    `hipcc` directly (mirrors `kiln-tensor/build.rs::build_rocm`). The `.cu`
//    sources `#include <cuda_runtime.h>` / `<cuda_bf16.h>`, which resolve to
//    the shared CUDA->HIP compat shim under `kiln-tensor/csrc/hip_compat`; the
//    four reduction kernels additionally `#include "kt_gpu_compat.cuh"` from
//    `kiln-tensor/csrc`. No-op (with a `cargo:warning`) when ROCm is absent.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    if cuda_enabled {
        build_cuda();
    }

    let rocm_enabled = env::var("CARGO_FEATURE_ROCM").is_ok();
    if rocm_enabled {
        build_rocm();
    }

    println!("cargo:rerun-if-changed=csrc/");
    println!("cargo:rerun-if-changed=build.rs");
}

// ---------------------------------------------------------------------
// CUDA path (unchanged behavior; now gated on CARGO_FEATURE_CUDA)
// ---------------------------------------------------------------------

fn build_cuda() {
    let cuda_root = match find_cuda_root() {
        Some(p) => p,
        None => {
            println!(
                "cargo:warning=CUDA not found, kiln-rmsnorm-kernel will not compile CUDA kernels"
            );
            println!("cargo:warning=Set CUDA_ROOT or CUDA_HOME, or install CUDA toolkit");
            return;
        }
    };

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc_dir = manifest_dir.join("csrc");

    let cuda_archs = env::var("KILN_CUDA_ARCHS").unwrap_or_else(|_| "80;86;89;90".to_string());

    let mut build = cc::Build::new();
    build.cuda(true);
    build.cpp(true);
    configure_nvcc_from_cuda_root(&cuda_root);

    build.include(&csrc_dir);
    build.include(cuda_root.join("include"));

    build.flag("-std=c++17");
    build.flag("-O3");
    build.flag("--use_fast_math");
    build.flag("--expt-relaxed-constexpr");
    build.flag("--expt-extended-lambda");
    build.flag("-Xcompiler").flag("-fPIC");
    build.flag("-diag-suppress=177");
    build.flag("-diag-suppress=174");

    for arch in cuda_archs.split(';') {
        let arch = arch.trim();
        if !arch.is_empty() {
            build.flag(&format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
        }
    }

    for kernel in KERNELS {
        build.file(csrc_dir.join(kernel));
    }

    build.compile("kiln_rmsnorm_kernel");

    println!(
        "cargo:rustc-link-search=native={}",
        cuda_root.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=cudart");

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
// ROCm / HIP kernel compile path (Phase R.7)
// ---------------------------------------------------------------------

/// The `csrc/*.cu` kernels, compiled identically by both the nvcc and the hipcc
/// path. The four reduction kernels (`fused_rmsnorm`, `fused_rmsnorm_bwd`,
/// `fused_l2_qk_norm`, `fused_attn_qkv_prep`) take the Phase R.7 wave-size fix
/// (`kt_gpu_compat.cuh` + `kiln_block_reduce_sum`, power-of-two blockDim>=64);
/// the rest are elementwise / per-thread and hipify-clean.
const KERNELS: &[&str] = &[
    "fused_rmsnorm.cu",
    "fused_rmsnorm_bwd.cu",
    "fused_l2_qk_norm.cu",
    "fused_rotary_qk.cu",
    "fused_attn_qkv_prep.cu",
    "fused_mlp_silu_mul.cu",
    "fused_sigmoid_mul.cu",
    "fused_lora_add.cu",
    "causal_conv1d_f32.cu",
    "optimizer_step.cu",
];

/// hipcc-compile the ROCm-side `csrc/*.cu` kernels into a static lib that the
/// Rust side links against. No-op (with a `cargo:warning`) when ROCm is absent,
/// so `cargo check --features rocm` stays green on a toolchain-less host.
///
/// The `.cu` sources `#include <cuda_runtime.h>` (+ `cuda_bf16.h`) and the
/// reduction kernels `#include "kt_gpu_compat.cuh"`. Both resolve under hipcc
/// via the shared shim in `kiln-tensor/csrc/hip_compat` + `kiln-tensor/csrc`,
/// so the sources compile byte-unchanged on both backends.
fn build_rocm() {
    let rocm_root = match find_rocm_root() {
        Some(p) => p,
        None => {
            println!(
                "cargo:warning=ROCm not found, kiln-rmsnorm-kernel's ROCm kernels will not be \
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

    // The shared CUDA->HIP compat shim + the wave-size helper header both live
    // under the kiln-tensor crate. Resolve them relative to THIS crate's
    // manifest dir (sibling crate under the same `crates/` parent).
    let kt_csrc = manifest_dir
        .parent()
        .expect("crate manifest dir has a parent")
        .join("kiln-tensor")
        .join("csrc");
    let kt_compat = kt_csrc.join("hip_compat");

    // gfx targets: CDNA (gfx90a/gfx942) + RDNA3 (gfx1100) per the v1 matrix,
    // plus gfx1151 (Strix Halo) so on-hardware parity testing runs on the dev
    // box. Override with KILN_ROCM_ARCHS.
    let archs =
        env::var("KILN_ROCM_ARCHS").unwrap_or_else(|_| "gfx942;gfx90a;gfx1100;gfx1151".to_string());

    let mut objects = Vec::new();
    for kernel in KERNELS {
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
    let lib = out_dir.join("libkiln_rmsnorm_kernel_rocm_ops.a");
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
    println!("cargo:rustc-link-lib=static=kiln_rmsnorm_kernel_rocm_ops");
    println!(
        "cargo:rustc-link-search=native={}",
        rocm_root.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    // hipcc device objects pull in the C++ runtime at final link.
    println!("cargo:rustc-link-lib=dylib=stdc++");

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
