// Build script for kiln-rocblas — the ROCm analog of kiln-blas/build.rs.
//
// Compiles the hipBLASLt host-side `.cu` wrappers with hipcc and links
// hipBLASLt + the HIP runtime. No-op (with a warning) when ROCm is absent, so
// `cargo check` works on a toolchain-less host. (`cc::Build::cuda(true)` is
// hardwired to nvcc, so we drive hipcc directly via Command — same as
// kiln-tensor/build.rs::build_rocm.)

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let want_probe = env::var_os("CARGO_FEATURE_PROBE").is_some();
    let want_hipblaslt = env::var_os("CARGO_FEATURE_HIPBLASLT").is_some();
    let want_primary_full = env::var_os("CARGO_PRIMARY_PACKAGE").is_some();

    if !want_probe && !want_hipblaslt && !want_primary_full {
        return;
    }

    let rocm_root = match find_rocm_root() {
        Some(p) => p,
        None => {
            println!(
                "cargo:warning=ROCm not found; kiln-rocblas hipBLASLt build skipped. Set \
                 ROCM_PATH / HIP_PATH or install ROCm to /opt/rocm."
            );
            return;
        }
    };

    let hipcc = env::var("HIPCC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| rocm_root.join("bin").join("hipcc"));
    if !hipcc.exists() {
        println!(
            "cargo:warning=hipcc not found at {}; skipping kiln-rocblas build.",
            hipcc.display()
        );
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc_dir = manifest_dir.join("csrc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // Local builds target the installed GPU. Release and cross builds can
    // request a semicolon-separated fat binary through KILN_ROCM_ARCHS.
    let archs = env::var("KILN_ROCM_ARCHS").unwrap_or_else(|_| "native".to_string());

    let mut sources: Vec<&str> = Vec::new();
    if want_probe || want_primary_full {
        sources.push("hipblaslt_probe.cu");
    }
    if want_hipblaslt || want_primary_full {
        sources.push("hipblaslt_matmul.cu");
    }

    let mut objects = Vec::new();
    for src_name in &sources {
        let src = csrc_dir.join(src_name);
        if !src.exists() {
            println!(
                "cargo:warning=kiln-rocblas: {} not found, skipping.",
                src.display()
            );
            continue;
        }
        let obj = out_dir.join(format!("{src_name}.o"));
        let mut cmd = Command::new(&hipcc);
        cmd.arg("-O3").arg("-std=c++17").arg("-fPIC").arg("-c");
        for arch in archs.split(';') {
            let a = arch.trim();
            if !a.is_empty() {
                cmd.arg(format!("--offload-arch={a}"));
            }
        }
        cmd.arg("-I").arg(&csrc_dir);
        cmd.arg("-I").arg(rocm_root.join("include"));
        cmd.arg(&src).arg("-o").arg(&obj);
        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("failed to spawn hipcc ({}): {e}", hipcc.display()));
        if !status.success() {
            panic!("hipcc failed to compile {src_name}");
        }
        objects.push(obj);
    }

    if objects.is_empty() {
        return;
    }

    let lib = out_dir.join("libkiln_rocblas.a");
    let _ = std::fs::remove_file(&lib);
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib);
    for o in &objects {
        ar.arg(o);
    }
    if !ar.status().expect("spawn ar").success() {
        panic!("ar failed to archive kiln-rocblas objects");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=kiln_rocblas");
    println!(
        "cargo:rustc-link-search=native={}",
        rocm_root.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=hipblaslt");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    println!("cargo:rerun-if-changed=csrc/");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=KILN_ROCM_ARCHS");
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
    None
}
