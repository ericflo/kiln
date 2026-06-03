// Build script for kiln-hip.
//
// Single responsibility: when a ROCm install is found, tell rustc to link
// `libamdhip64.so` (the HIP runtime). When it is not found, emit a warning and
// link nothing — the crate still *compiles* (the FFI block has no `#[link]`
// attribute), so `cargo check` succeeds on any host. Only a final binary/test
// that actually calls the HIP functions needs the library at link time, and
// that only happens on a ROCm host building `--features rocm`.
//
// This mirrors `kiln-tensor/build.rs::find_cuda_root` for the ROCm toolchain.

use std::env;
use std::path::PathBuf;

/// Locate a ROCm install root containing `lib/libamdhip64.so`.
///
/// Honours `ROCM_PATH` / `HIP_PATH`, then falls back to `/opt/rocm`.
fn find_rocm_root() -> Option<PathBuf> {
    for var in ["ROCM_PATH", "HIP_PATH"] {
        if let Ok(p) = env::var(var) {
            let root = PathBuf::from(p);
            if root.join("lib").is_dir() {
                return Some(root);
            }
        }
    }
    let default = PathBuf::from("/opt/rocm");
    if default.join("lib").is_dir() {
        return Some(default);
    }
    None
}

fn main() {
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");

    match find_rocm_root() {
        Some(root) => {
            println!(
                "cargo:rustc-link-search=native={}",
                root.join("lib").display()
            );
            // The HIP runtime. `dylib` because ROCm ships shared objects.
            println!("cargo:rustc-link-lib=dylib=amdhip64");
        }
        None => {
            println!(
                "cargo:warning=kiln-hip: ROCm not found (set ROCM_PATH/HIP_PATH or install to \
                 /opt/rocm). libamdhip64 will not be linked; `cargo check` still works, but \
                 building or testing a ROCm binary requires a ROCm toolchain."
            );
        }
    }
}
