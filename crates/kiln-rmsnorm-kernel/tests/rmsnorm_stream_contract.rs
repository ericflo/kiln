//! Cheap source contract for RMSNorm's ROCm stream/context handoff.

const KT_API: &str = include_str!("../src/kt_api.rs");

fn function_body(name: &str, next_name: &str) -> &'static str {
    KT_API
        .split_once(name)
        .unwrap_or_else(|| panic!("missing function {name}"))
        .1
        .split_once(next_name)
        .unwrap_or_else(|| panic!("missing boundary {next_name}"))
        .0
}

#[test]
fn backward_preserves_rocm_context_and_orders_only_cross_stream_inputs() {
    let allocator = function_body(
        "fn alloc_rmsnorm_backward_like(",
        "fn device_stream_submission(",
    );
    assert!(allocator.contains("rocm_storage_and_byte_offset"));
    assert!(allocator.contains("alloc_rocm_tensor(storage, dtype, shape)"));

    let handoff = function_body(
        "fn synchronize_rocm_rmsnorm_backward_inputs(",
        "/// `fused_rmsnorm`",
    );
    assert!(handoff.contains("rocm_owner_stream_identity(tensor, name)"));
    assert!(handoff.contains("input_owner_stream != output_owner_stream"));
    assert!(
        handoff.find("if capture_active").expect("capture check")
            < handoff
                .find("rocm_active_stream_identity(tensor, name)")
                .expect("active stream lookup")
    );
    assert!(handoff.contains("if input_stream == launch_stream"));
    assert!(handoff.contains("rocm_synchronize_tensor_stream(tensor)"));
    assert!(handoff.contains("rocm_capture_arena_active()"));

    let backward = function_body("fn fused_rmsnorm_backward_impl(", "/// `fused_rotary_qk`");
    assert!(backward.contains("alloc_rmsnorm_backward_like"));
    assert!(backward.contains("output_owner_stream != x_owner_stream"));
    assert!(backward.contains("synchronize_rocm_rmsnorm_backward_inputs"));
    assert!(!backward.contains("rocm_synchronize_default_stream"));
    assert!(!backward.contains("hipDeviceSynchronize"));
}
