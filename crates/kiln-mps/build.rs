use std::env;

fn main() {
    // Only link Metal frameworks when the `probe` feature is on AND
    // we're targeting Apple (Metal + MetalKit + Foundation are Apple-
    // only, and `library kind 'framework' is only supported on Apple
    // targets`). The backend-agnostic types compile on every host
    // without `--features probe`; with `--features probe` on a non-
    // Apple host we still want the crate to type-check cleanly so the
    // Phase 2.2 backend-agnostic types stay buildable everywhere —
    // the actual Metal-runtime probe binary lives behind a separate
    // Apple-only cfg and pulls metal-rs directly when it lands.
    //
    // #1082: previously this just unconditionally emitted the framework
    // link lines whenever CARGO_FEATURE_PROBE was set, which made
    // `cargo check -p kiln-mps --features probe` fail on Linux
    // (rustc rejects `framework=*` on non-Apple targets). Gate on
    // CARGO_CFG_TARGET_OS / CARGO_CFG_TARGET_VENDOR to mirror cargo's
    // own platform detection.
    let want_probe = env::var_os("CARGO_FEATURE_PROBE").is_some();
    let target_vendor = env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_apple = target_vendor == "apple"
        || matches!(
            target_os.as_str(),
            "macos" | "ios" | "tvos" | "watchos" | "visionos"
        );
    if want_probe && is_apple {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PROBE");
}
