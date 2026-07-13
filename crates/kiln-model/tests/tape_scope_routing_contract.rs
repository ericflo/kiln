//! CPU-only source contracts for tape-authoritative model routing.

const FORWARD: &str = include_str!("../src/forward.rs");
const TAPE_FORWARD: &str = include_str!("../src/tape_forward.rs");

fn section(start: &str, end: &str) -> &'static str {
    let (_, tail) = FORWARD
        .split_once(start)
        .unwrap_or_else(|| panic!("missing section start: {start}"));
    let (body, _) = tail
        .split_once(end)
        .unwrap_or_else(|| panic!("missing section end after {start}: {end}"));
    body
}

fn assert_before(body: &str, first: &str, second: &str) {
    let first_at = body
        .find(first)
        .unwrap_or_else(|| panic!("missing first marker: {first}"));
    let second_at = body
        .find(second)
        .unwrap_or_else(|| panic!("missing second marker: {second}"));
    assert!(first_at < second_at, "expected `{first}` before `{second}`");
}

fn assert_required_recorder(body: &str, recorder: &str, operation: &str) {
    assert!(
        body.contains(recorder),
        "missing recorder `{recorder}` for {operation}"
    );
    assert!(
        body.contains("require_active_tape_output"),
        "{operation} must convert a recorder decline into an error"
    );
    assert!(
        body.contains(operation),
        "{operation} must have a stable fail-closed diagnostic"
    );
}

#[test]
fn fused_rope_leaves_are_inference_only() {
    let from_tensor = section(
        "pub fn rotary_embedding_from_tensor(",
        "pub(crate) fn rotary_tables_from_tensor(",
    );
    assert_eq!(
        from_tensor
            .matches("!crate::tape_forward::tape_scope_active()")
            .count(),
        2,
        "CUDA and ROCm fused RoPE leaves must both reject active tape scopes"
    );
    assert!(from_tensor.contains("let rotated_q = apply_rope("));
    assert!(from_tensor.contains("let rotated_k = apply_rope("));

    let from_tables = section(
        "fn rotary_embedding_from_tables(",
        "/// Residual add `a + b`",
    );
    assert_eq!(
        from_tables
            .matches("!crate::tape_forward::tape_scope_active()")
            .count(),
        3,
        "CUDA, ROCm, and Metal fused RoPE leaves must reject active tape scopes"
    );
    assert!(from_tables.contains("let rotated_q = apply_rope("));
    assert!(from_tables.contains("let rotated_k = apply_rope("));

    let apply = section("fn apply_rope(", "pub(crate) fn try_kt_concat_last_dim(");
    assert!(
        apply.contains(".context(\"active tape scope could not record RoPE\")?"),
        "active-scope RoPE must fail closed when its recorder declines"
    );
    assert_before(
        apply,
        "if crate::tape_forward::tape_scope_active()",
        "let half_rotary = rotary_dim / 2",
    );
}

#[test]
fn metal_swiglu_records_before_forward_only_leaves() {
    let gated_hidden = section(
        "pub fn swiglu_ffn_gated_hidden(",
        "/// SwiGLU down projection half",
    );
    assert_before(
        gated_hidden,
        "try_tape_swiglu_kt(&gate, &up)",
        "metal_mlp_silu_mul_bf16(&gate, &up)",
    );
    assert!(gated_hidden.contains(
        "if !crate::tape_forward::tape_scope_active()\n            && crate::backend::metal::metal_mlp_silu_mul_supports"
    ));

    let no_chunk = section("fn swiglu_ffn_impl_no_chunk(", "fn swiglu_ffn_b12_tapped(");
    assert!(no_chunk.contains("let tape_scope_active = crate::tape_forward::tape_scope_active();"));
    assert!(
        no_chunk.matches("if !tape_scope_active").count() >= 4,
        "whole-MLP, gate-up, packed, and Metal leaves must remain inference-only"
    );
    assert_before(
        no_chunk,
        "try_tape_swiglu_kt(&gate, &up)",
        "metal_mlp_silu_mul_bf16(&gate, &up)",
    );
    assert!(no_chunk.contains(".context(\"active tape scope failed to record SwiGLU\")?"));
    assert!(
        no_chunk.contains(".context(\"active tape scope failed to record Vulkan SwiGLU\")?"),
        "Vulkan must fail closed if its active-scope recorder disappears"
    );
}

#[test]
fn rmsnorm_scope_precedes_switches_and_fails_closed_on_every_gpu() {
    let rms_norm = section(
        "pub fn rms_norm(",
        "fn rocm_fused_rmsnorm_allowed_for_tensor(",
    );
    let scope = "if crate::tape_forward::tape_scope_active()";
    let failure = "active tape scope could not record RMSNorm";
    assert_eq!(
        rms_norm.matches(scope).count(),
        1,
        "RMSNorm must have one backend-independent tape-routing authority"
    );
    assert!(rms_norm.contains(failure), "RMSNorm must fail closed");
    assert_before(rms_norm, scope, failure);

    for inference_marker in [
        "let kernel_disabled = std::env::var(\"KILN_DISABLE_RMSNORM_KERNEL\")",
        "metal_rms_norm_bf16",
        "try_vulkan_rmsnorm_forward",
        "// ROCm inference path",
    ] {
        assert_before(rms_norm, failure, inference_marker);
    }
}

#[test]
fn primitive_and_projection_routes_fail_closed_before_inference() {
    let silu = section("fn cuda_silu(", "fn try_kt_silu_composite(");
    assert_required_recorder(silu, "try_tape_silu_kt", "SiLU");
    assert_before(silu, "try_tape_silu_kt", "try_kt_silu_composite");
    for feature in ["cuda", "metal", "vulkan", "rocm"] {
        assert!(
            silu.contains(&format!("feature = \"{feature}\"")),
            "SiLU scope-first routing must compile for {feature}"
        );
    }

    let lora_add = section(
        "fn add_lora_delta_to_base(",
        "fn linear_with_lora_t_decode_if(",
    );
    assert_required_recorder(lora_add, "try_tape_lora_add_kt", "LoRA delta add");
    assert_before(lora_add, "try_tape_lora_add_kt", "try_kt_lora_delta");

    let linear = section(
        "fn linear_with_lora_t_backend_decode_if(",
        "fn metal_attn_gate_debug_active(",
    );
    assert_required_recorder(linear, "try_tape_lora_linear_kt", "linear with LoRA");
    assert_before(
        linear,
        "try_tape_lora_linear_kt",
        "if let Some(backend) = backend",
    );

    let gate = section(
        "fn attention_output_gate_decode_if(",
        "fn full_attn_qkv_proj_decode_if(",
    );
    assert_required_recorder(
        gate,
        "try_tape_attn_gate_sigmoid_mul_kt",
        "attention output gate",
    );
    assert_before(
        gate,
        "try_tape_attn_gate_sigmoid_mul_kt",
        "metal_attn_gate_sigmoid_mul_bf16",
    );

    let matmul = section("pub(crate) fn try_kt_matmul(", "fn gdn_in_proj_matmul(");
    assert_required_recorder(matmul, "try_tape_matmul_kt", "matmul");
    assert_before(matmul, "try_tape_matmul_kt", "kiln_tensor::ops::matmul");

    let gdn_in = section("fn gdn_in_proj_matmul(", "fn promote_cpu_activation(");
    assert_required_recorder(gdn_in, "try_tape_lora_linear_kt", "GDN input projection");
    assert_before(gdn_in, "try_tape_lora_linear_kt", "runtime_matmul");

    let embeddings = section("pub fn embedding_lookup(", "fn kt_embedding_lookup_native(");
    for (recorder, operation) in [
        ("try_tape_frozen_embedding_kt", "token embedding lookup"),
        (
            "try_tape_frozen_embedding_kt",
            "indexed token embedding lookup",
        ),
    ] {
        assert_required_recorder(embeddings, recorder, operation);
    }

    let residual = section("fn residual_add(", "fn apply_rope(");
    assert_required_recorder(residual, "try_tape_add_kt", "residual add");

    let lm_head = section(
        "fn lm_head_forward_backend_decode_if(",
        "fn lm_head_argmax_with_backend(",
    );
    assert_required_recorder(lm_head, "try_tape_lora_linear_kt", "LM head projection");
}

#[test]
fn split_lora_records_original_parameter_before_inference_slicing() {
    let split = section(
        "fn linear_with_lora_t_backend_decode_output_slice(",
        "fn split_q_gate_training_bf16(",
    );
    assert_required_recorder(
        split,
        "try_tape_lora_linear_output_slice_kt",
        "split query/gate projection",
    );
    assert_before(
        split,
        "try_tape_lora_linear_output_slice_kt",
        "lora_projection_slice",
    );

    let split_outputs = section(
        "fn split_q_gate_training_bf16(",
        "pub struct GqaAttentionPrepared",
    );
    assert_eq!(
        split_outputs.matches("tape_reshape_full_attn(").count(),
        2,
        "split q/gate outputs must both retain their projection lineage"
    );
    assert!(!split_outputs.contains("reshape_hole0_4(&q_flat"));
    assert!(!split_outputs.contains("reshape_hole0_3(&gate"));
}

#[test]
fn active_scope_rejects_projection_fast_leaves() {
    let qkv = section(
        "fn full_attn_qkv_proj_decode_if(",
        "/// CUDA-compatible softmax on last dimension.",
    );
    assert!(qkv.contains("if !tape_scope_active"));
    assert_before(qkv, "if !tape_scope_active", "runtime_full_attn_qkv_decode");

    let mlp = section("fn mlp_proj_forward_decode_if(", "fn lm_head_forward(");
    assert!(mlp.contains("!crate::tape_forward::tape_scope_active()"));
    assert!(mlp.contains("active tape scope cannot use forward-only Marlin MLP projection"));
    assert_before(
        mlp,
        "!crate::tape_forward::tape_scope_active()",
        "marlin_proj::matmul_bf16_kt",
    );

    let q_proj = section(
        "fn q_proj_forward_decode_if(",
        "fn split_q_gate_training_disabled(",
    );
    assert!(q_proj.contains("!crate::tape_forward::tape_scope_active()"));
    assert!(q_proj.contains("active tape scope cannot use forward-only Marlin query projection"));
    assert_before(
        q_proj,
        "!crate::tape_forward::tape_scope_active()",
        "marlin_proj::matmul_bf16_kt",
    );

    let output_projection = section(
        "pub fn gqa_attention_output_projection(",
        "/// Returns: [batch, seq_len, hidden_size]",
    );
    assert!(output_projection.contains("!crate::tape_forward::tape_scope_active()"));
    assert_before(
        output_projection,
        "!crate::tape_forward::tape_scope_active()",
        "o_proj_w8.as_ref()",
    );
}

#[test]
fn gdn_terminal_recorders_are_required() {
    let gates = section(
        "fn gated_deltanet_gates_fallback(",
        "pub fn gated_deltanet_forward(",
    );
    assert_required_recorder(gates, "try_tape_gdn_gates_kt", "GDN gate transforms");

    let qk_norm = section("fn gdn_qk_norm(", "fn gdn_qk_norm_forward(");
    assert_required_recorder(
        qk_norm,
        "try_tape_gdn_l2_norm_scale_kt",
        "GDN query L2 normalization",
    );
    assert!(qk_norm.contains("GDN key L2 normalization"));

    let gdn = section(
        "fn gated_deltanet_forward_decode_if_inner(",
        "/// Grouped-Query Attention (GQA).",
    );
    for operation in [
        "GDN single-token conv input reshape",
        "GDN conv input transpose",
        "GDN causal conv1d prefill",
        "GDN conv output transpose",
        "GDN QKV narrow",
        "GDN QKV reshape",
        "GDN z reshape",
        "GDN query GQA expansion",
        "GDN key GQA expansion",
        "GDN recurrent query GQA expansion",
        "GDN recurrent key GQA expansion",
        "GDN recurrent input transpose",
        "GDN recurrence output transpose",
        "GDN gated RMSNorm",
        "GDN gated-norm output reshape",
        "GDN gated-norm output cast",
    ] {
        assert!(
            gdn.contains(operation),
            "GDN route lacks fail-closed marker for {operation}"
        );
    }
    assert!(gdn.contains("active tape scope could not record GDN recurrence"));
    assert!(gdn.contains("active tape scope cannot run GDN single-token causal conv1d"));

    // The fused conv+SiLU strategy may decline, but the alternate must restore
    // state, require the standalone conv recorder, then use scope-first SiLU.
    assert_before(
        gdn,
        "*conv_state = conv_entry_state",
        "record_prefill_conv(&y)",
    );
    assert_before(gdn, "record_prefill_conv(&y)", "cuda_silu(&y)");

    let split = section(
        "fn linear_with_lora_t_backend_decode_output_slice(",
        "fn split_q_gate_training_bf16(",
    );
    assert_required_recorder(
        split,
        "try_tape_lora_linear_output_slice_kt",
        "split query/gate projection",
    );
    assert_required_recorder(
        split,
        "try_tape_concat_kt",
        "split query/gate projection concatenation",
    );
}

#[test]
fn gdn_gate_recorder_exposes_only_projection_activations() {
    let (_, tail) = TAPE_FORWARD
        .split_once("pub fn try_tape_gdn_gates_kt(")
        .expect("missing GDN gate tape recorder");
    let (recorder, _) = tail
        .split_once("/// Tape backward for the GDN")
        .expect("missing end of GDN gate tape recorder");
    assert!(recorder.contains("beta,\n            &[b],"));
    assert!(recorder.contains("g,\n            &[a],"));
    assert!(!recorder.contains("&[a, a_log"));
    assert!(!recorder.contains("&[b, dt_bias"));
    assert!(recorder.contains("active tape scope disappeared while recording GDN gates"));
}

#[test]
fn gdn_gated_rmsnorm_fused_backward_treats_weight_as_frozen() {
    let (_, tail) = TAPE_FORWARD
        .split_once("fn try_gdn_gated_rms_norm_backward_fused_cuda_rocm(")
        .expect("missing fused GDN gated RMSNorm backward");
    let (fused_backward, _) = tail
        .split_once("impl BackwardOp for GdnGatedRmsNormBackward")
        .expect("missing end of fused GDN gated RMSNorm backward");

    for frozen_api in [
        "gdn_gated_rms_norm_bwd_bf16_frozen_weight_kt(",
        "gdn_gated_rms_norm_bwd_bf16_f32_weight_frozen_kt(",
    ] {
        assert!(
            fused_backward.contains(frozen_api),
            "fused model backward must use `{frozen_api}`"
        );
    }
    for trainable_api in [
        "gdn_gated_rms_norm_bwd_bf16_kt(",
        "gdn_gated_rms_norm_bwd_bf16_f32_weight_kt(",
    ] {
        assert!(
            !fused_backward.contains(trainable_api),
            "frozen model backward must not use `{trainable_api}`"
        );
    }
}

#[test]
fn gdn_gated_rmsnorm_portable_backward_treats_weight_as_frozen() {
    let (_, tail) = TAPE_FORWARD
        .split_once("impl BackwardOp for GdnGatedRmsNormBackward")
        .expect("missing GDN gated RMSNorm backward op");
    let (backward, _) = tail
        .split_once("/// Route the GDN gated RMSNorm")
        .expect("missing end of GDN gated RMSNorm backward op");

    assert!(backward.contains("gdn_gated_rms_norm_frozen_weight_backward_no_grad("));
    assert!(
        !backward.contains("gdn_gated_rms_norm_backward_no_grad("),
        "portable model backward must not call a dw-producing helper"
    );
    assert!(
        !backward.contains("grads.dw"),
        "portable model backward must not compute and discard a frozen weight gradient"
    );
}

#[test]
fn full_attention_terminal_routes_require_every_recorder() {
    let prefill = section(
        "pub fn gqa_attention_core_prefill(",
        "pub fn gqa_attention_apply_output_gate(",
    );
    for operation in [
        "GQA flash-attention output reshape",
        "GQA SDPA fallback",
        "GQA SDPA output transpose",
        "GQA SDPA output reshape",
    ] {
        assert!(prefill.contains(operation), "missing {operation}");
    }
    assert_before(
        prefill,
        "try_tape_flash_attn_kt",
        "try_tape_sdpa_fallback_kt",
    );
    assert_before(
        prefill,
        "try_tape_sdpa_fallback_kt",
        "GQA SDPA output reshape",
    );

    let full = section(
        "pub fn gqa_attention_pre_o(",
        "pub fn gqa_attention_output_projection(",
    );
    for operation in [
        "full-attention flash output reshape",
        "active tape scope requires pre-expand SDPA inputs",
        "full-attention SDPA fallback",
        "full-attention SDPA output transpose",
        "full-attention SDPA output reshape",
    ] {
        assert!(full.contains(operation), "missing {operation}");
    }
    assert_before(full, "try_tape_flash_attn_kt", "try_tape_sdpa_fallback_kt");

    for helper in [
        section(
            "fn tape_reshape_full_attn(",
            "fn tape_narrow_contig_full_attn(",
        ),
        section(
            "fn tape_transpose_contig_full_attn(",
            "/// A single axis spec for [`tape_reshape_full_attn`]",
        ),
    ] {
        assert!(helper.contains("require_active_tape_output"));
    }

    let chunked = section(
        "fn transformer_block_detached_prefill_chunked(",
        "fn transformer_block_paged_with_rope_tables(",
    );
    assert!(chunked.contains("gqa_attention_core_prefill("));
    assert!(chunked.contains("tape replay failed to record output cat"));
}

#[test]
fn unused_prefill_helpers_fail_closed_in_training_scope() {
    for (start, end, diagnostic) in [
        (
            "pub fn gqa_attention_q_gate_prefill(",
            "pub fn gqa_attention_kv_prefill(",
            "gqa_attention_q_gate_prefill is inference-only",
        ),
        (
            "pub fn gqa_attention_kv_prefill(",
            "pub fn gqa_attention_prepare_prefill(",
            "gqa_attention_kv_prefill is inference-only",
        ),
        (
            "pub fn gqa_attention_prepare_prefill(",
            "pub fn gqa_attention_core_prefill(",
            "gqa_attention_prepare_prefill is inference-only",
        ),
        (
            "pub fn gqa_attention_pre_o_chunked_prefill(",
            "pub fn gqa_attention_pre_o(",
            "gqa_attention_pre_o_chunked_prefill is inference-only",
        ),
    ] {
        let helper = section(start, end);
        assert!(
            helper.contains("tape_scope_active()"),
            "missing scope guard for {start}"
        );
        assert!(
            helper.contains(diagnostic),
            "missing stable diagnostic for {start}"
        );
        for unsafe_operation in ["reshape_hole0_", ".narrow(", "Tensor::cat"] {
            if helper.contains(unsafe_operation) {
                assert_before(helper, diagnostic, unsafe_operation);
            }
        }
    }
}

#[test]
fn flash_recorder_admits_only_one_exact_cuda_or_rocm_device() {
    let flash = {
        let (_, tail) = TAPE_FORWARD
            .split_once("pub fn try_tape_flash_attn_kt(")
            .expect("missing flash tape recorder");
        let (body, _) = tail
            .split_once("/// Offload saved GDN recurrence tensors")
            .expect("missing flash recorder end marker");
        body
    };
    assert!(!flash.contains("kiln_tensor::Device::Metal"));
    assert!(flash.contains("kiln_tensor::Device::Cuda"));
    assert!(flash.contains("kiln_tensor::Device::Rocm"));
    assert!(flash.contains("k.device() != q.device()"));
    assert!(flash.contains("v.device() != q.device()"));
}
