use std::env;

fn main() {
    // Only link Metal frameworks when the `probe` feature is on;
    // backend-agnostic types compile without them.
    if env::var_os("CARGO_FEATURE_PROBE").is_some() {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PROBE");
}
