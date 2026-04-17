// Link against libnvToolsExt only when the `nvtx` feature is enabled.
// The library ships with the CUDA toolkit at $CUDA_HOME/lib64.
//
// When the feature is off this build script is a no-op so non-CUDA / non-nsys
// builds (e.g. CPU-only CI, training-only laptops) do not need any CUDA env.

fn main() {
    if std::env::var("CARGO_FEATURE_NVTX").is_ok() {
        let cuda_home = std::env::var("CUDA_HOME")
            .or_else(|_| std::env::var("CUDA_ROOT"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        println!("cargo:rustc-link-search=native={cuda_home}/lib64");
        println!("cargo:rustc-link-lib=dylib=nvToolsExt");
    }
}
