//! Static contract for the adapter-training GDN gated RMSNorm backward.
//!
//! Numerical parity is covered on CUDA and ROCm by the backend tests. These
//! checks pin the resource-ownership property that numerical tests cannot see:
//! frozen normalization weights must not allocate or atomically reduce dWeight.

const CUDA_SOURCE: &str = include_str!("../csrc/gdn_gated_rms_norm.cu");
const C_HEADER: &str = include_str!("../csrc/gdn_gated_rms_norm.h");
const KT_API: &str = include_str!("../src/kt_api.rs");
const LIB_RS: &str = include_str!("../src/lib.rs");

fn declaration_from<'a>(source: &'a str, symbol: &str) -> &'a str {
    let start = source
        .find(symbol)
        .unwrap_or_else(|| panic!("missing declaration for {symbol}"));
    let tail = &source[start..];
    let end = tail
        .find(");")
        .unwrap_or_else(|| panic!("unterminated declaration for {symbol}"));
    &tail[..end + 2]
}

#[test]
fn frozen_kernel_instantiations_compile_out_dweight_atomics() {
    assert!(CUDA_SOURCE.contains("template <typename WeightT, bool ComputeWeightGrad>"));
    assert!(CUDA_SOURCE.contains("if constexpr (ComputeWeightGrad)"));
    assert_eq!(
        CUDA_SOURCE
            .matches("atomicAdd(&d_weight[tid], dw);")
            .count(),
        1,
        "dWeight has one guarded atomic site"
    );
    assert!(CUDA_SOURCE.contains("launch_gdn_gated_rms_norm_bwd<__nv_bfloat16, false>"));
    assert!(CUDA_SOURCE.contains("launch_gdn_gated_rms_norm_bwd<float, false>"));
}

#[test]
fn frozen_ffi_symbols_have_no_dweight_parameter() {
    for symbol in [
        "kiln_gdn_gated_rms_norm_bwd_frozen_bf16(",
        "kiln_gdn_gated_rms_norm_bwd_frozen_wf32_bf16(",
    ] {
        let declaration = declaration_from(C_HEADER, symbol);
        assert!(declaration.contains("void* d_x"));
        assert!(declaration.contains("void* d_z"));
        assert!(
            !declaration.contains("d_weight"),
            "{symbol} must not accept a dWeight buffer"
        );
    }
}

#[test]
fn rust_api_exposes_activation_only_results_for_both_weight_dtypes() {
    let result_start = KT_API
        .find("pub struct GdnGatedRmsNormFrozenWeightBwdKt")
        .expect("frozen result type");
    let result_tail = &KT_API[result_start..];
    let result_end = result_tail.find('}').expect("frozen result terminator");
    let result = &result_tail[..=result_end];
    assert!(result.contains("pub dx: KtTensor"));
    assert!(result.contains("pub dz: KtTensor"));
    assert!(!result.contains("dw"));

    for api in [
        "gdn_gated_rms_norm_bwd_bf16_frozen_weight_kt",
        "gdn_gated_rms_norm_bwd_bf16_f32_weight_frozen_kt",
    ] {
        assert!(KT_API.contains(&format!("pub fn {api}(")));
        assert!(
            LIB_RS.contains(api),
            "{api} must be exported from the crate"
        );
    }
    assert!(KT_API.contains("let dw = if compute_weight_grad"));
}
