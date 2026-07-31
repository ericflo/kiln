#!/usr/bin/env python3
"""Generate Kiln's closed read-only serving and observability JSON Schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_DIR = ROOT / "scripts" / "qualification"
if str(QUALIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(QUALIFICATION_DIR))

from request_latency_contract import (  # noqa: E402
    LATENCY_PHASE_FIELDS,
    LATENCY_STALL_REASON_FIELDS,
)

OUTPUT = ROOT / "contracts" / "kiln-observability-v1.schema.json"
STATUS = {"x-kiln-field-schema-status": "complete"}
DEFS: dict[str, dict[str, Any]] = {}


def ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/$defs/{name}"}


def nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def array(
    schema: dict[str, Any],
    *,
    min_items: int | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {"type": "array", "items": schema}
    if min_items is not None:
        value["minItems"] = min_items
    if max_items is not None:
        value["maxItems"] = max_items
    return value


def mapping(schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": schema}


def described(schema: dict[str, Any], description: str) -> dict[str, Any]:
    return {**schema, "description": description}


def add_definition(name: str, rust_type: str, schema: dict[str, Any], description: str) -> None:
    DEFS[name] = {
        **schema,
        "description": description,
        "x-kiln-rust-type": rust_type,
        **STATUS,
    }


def add_enum(name: str, rust_type: str, values: list[str], description: str) -> None:
    add_definition(name, rust_type, {"type": "string", "enum": values}, description)


def add_object(
    name: str,
    rust_type: str,
    fields: dict[str, dict[str, Any]],
    description: str,
    *,
    optional: tuple[str, ...] = (),
) -> None:
    optional_set = set(optional)
    unknown = optional_set - set(fields)
    if unknown:
        raise ValueError(f"{name}: optional fields are not declared: {sorted(unknown)}")
    add_definition(
        name,
        rust_type,
        {
            "type": "object",
            "additionalProperties": False,
            "required": [field for field in fields if field not in optional_set],
            "properties": fields,
        },
        description,
    )


FIELD_DESCRIPTIONS = {
    ("AcceleratorWeightUploadConfigResponse", "configured_mib_per_second"):
        "Configured accelerator-upload rate in MiB per second; null disables pacing.",
    ("AcceleratorWeightUploadConfigResponse", "not_applicable_reason"):
        "Why upload pacing does not apply; null when the policy applies.",
    ("ConfigResponse", "rocm_graphs"):
        "Point-in-time ROCm graph statistics, or null when a nonblocking snapshot is unavailable.",
    ("ConfigResponse", "rocm_graphs_unavailable_reason"):
        "Closed reason `rocm_graphs` is null; null when statistics are present.",
    ("ConfigResponse", "rocm_graph_telemetry"):
        "Lock-independent ROCm graph phase telemetry, or null when no graph runner exists.",
    ("ConfigResponse", "rocm_graph_telemetry_unavailable_reason"):
        "Closed reason `rocm_graph_telemetry` is null; null when telemetry is present.",
    ("CudaGraphInfo", "enabled"):
        "Whether CUDA graph replay is enabled; null while the nonblocking snapshot is busy.",
    ("DebugDisabledResponse", "error"):
        "Stable error code for a disabled debug endpoint.",
    ("DebugDisabledResponse", "enable_with"):
        "Configuration change required before restart to enable the endpoint.",
    ("DebugProvenanceErrorResponse", "error"):
        "Stable error code for resident provenance that failed validation.",
    ("DecodeStatsSnapshot", "window_secs"):
        "Fixed rolling window in seconds.",
    ("GenerationConfig", "default_thinking_enabled"):
        "Server-wide thinking default; null preserves the model template's default.",
    ("GenerationConfig", "default_thinking_budget_tokens"):
        "Default thinking-token limit; null means unlimited.",
    ("GenerationConfig", "default_thinking_budget_ms"):
        "Default thinking-time limit in milliseconds; null means unlimited.",
    ("HealthResponse", "status"):
        "`ok` is ready for normal work, `degraded` reports a failed readiness check, and "
        "`maintenance` reports an intentionally drained server.",
    ("HealthResponse", "backend"):
        "Whether the server is using mock responses or a loaded model.",
    ("HealthResponse", "default_thinking_enabled"):
        "Server-wide thinking default; null preserves the model template's default.",
    ("HealthResponse", "default_thinking_budget_tokens"):
        "Default thinking-token limit; null means unlimited.",
    ("HealthResponse", "default_thinking_budget_ms"):
        "Default thinking-time limit in milliseconds; null means unlimited.",
    ("HealthResponse", "base_weight_identity"):
        "Loaded base-weight identity; null in mock mode.",
    ("HealthResponse", "execution_identity"):
        "Validated execution identity; null when no real-model provenance is resident.",
    ("HealthResponse", "active_adapter"):
        "Adapter selected for new inference requests; null selects the base model.",
    ("HealthResponse", "loaded_adapter"):
        "Adapter currently resident in the model runner; null when only base weights are loaded.",
    ("HealthResponse", "loaded_adapter_revision"):
        "Content revision of the resident adapter; null when no adapter is loaded.",
    ("HealthResponse", "scheduler"):
        "Paged-KV scheduler gauges; null when the active backend does not expose them.",
    ("HealthResponse", "gpu_memory"):
        "GPU capacity and live all-process memory observation; null without a GPU backend.",
    ("HealthResponse", "checks"):
        "The six stable readiness checks whose failures determine `degraded` status and HTTP 503.",
    ("HttpRuntimeInfo", "send_buffer_requested_bytes"):
        "Configured send-buffer target in bytes, or null when Kiln leaves the socket default unchanged.",
    ("HttpRuntimeInfo", "send_buffer_kernel_readback_bytes"):
        "Send-buffer size reported by the operating system after listener setup, or null when no target "
        "was requested.",
    ("HttpRuntimeInfo", "send_buffer_effective_bytes"):
        "Effective send-buffer target used for startup validation, or null when Kiln leaves the socket "
        "default unchanged.",
    ("ModelInfo", "id"):
        "Served model ID accepted by the inference API.",
    ("ModelInfo", "object"):
        "OpenAI-compatible model discriminator.",
    ("ModelInfo", "owned_by"):
        "Stable owner label for a model served by Kiln.",
    ("ModelStateResponse", "rocm_graphs"):
        "Point-in-time ROCm graph statistics, or null when a nonblocking snapshot is unavailable.",
    ("ModelStateResponse", "rocm_graphs_unavailable_reason"):
        "Closed reason `rocm_graphs` is null; null when statistics are present.",
    ("ModelStateResponse", "rocm_graph_telemetry"):
        "Lock-independent ROCm graph phase telemetry, or null when no graph runner exists.",
    ("ModelStateResponse", "rocm_graph_telemetry_unavailable_reason"):
        "Closed reason `rocm_graph_telemetry` is null; null when telemetry is present.",
    ("ModelsResponse", "object"):
        "OpenAI-compatible list discriminator.",
    ("ModelsResponse", "data"):
        "The single model served by this Kiln process.",
}


def apply_editorial_descriptions() -> None:
    for (definition_name, field_name), description in FIELD_DESCRIPTIONS.items():
        definition = DEFS.get(definition_name)
        properties = definition.get("properties") if definition else None
        if not isinstance(properties, dict) or field_name not in properties:
            raise ValueError(
                f"editorial description has no generated field owner: "
                f"{definition_name}.{field_name}"
            )
        properties[field_name]["description"] = description


def build_definitions() -> None:
    add_definition("Boolean", "bool", {"type": "boolean"}, "A serialized Rust boolean.")
    add_definition("String", "String", {"type": "string"}, "A serialized UTF-8 Rust string.")
    add_definition(
        "NonEmptyString",
        "String",
        {"type": "string", "minLength": 1},
        "A non-empty serialized UTF-8 Rust string.",
    )
    add_definition(
        "NonNegativeInteger",
        "u64 | u32 | usize",
        {"type": "integer", "minimum": 0},
        "A non-negative Rust integer represented exactly as a JSON number.",
    )
    add_definition(
        "PositiveInteger",
        "NonZeroUsize",
        {"type": "integer", "minimum": 1},
        "A positive Rust integer represented exactly as a JSON number.",
    )
    add_definition(
        "FiniteNumber",
        "f32 | f64",
        {"type": "number"},
        "A finite Rust floating-point value. serde_json rejects non-finite values.",
    )
    add_definition(
        "NonNegativeNumber",
        "f32 | f64",
        {"type": "number", "minimum": 0},
        "A finite non-negative Rust floating-point value.",
    )
    add_definition(
        "Sha256",
        "String",
        {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
        "A lowercase SHA-256 digest with Kiln's sha256: prefix.",
    )

    add_enum(
        "ConfigValueSource",
        "ConfigValueSource",
        ["default", "config_file", "environment", "command_line"],
        "The startup authority that supplied a resolved configuration value.",
    )
    add_enum(
        "ServingProfile",
        "ServingProfile",
        ["stable", "experimental", "maintenance"],
        "The immutable process-lifetime serving profile.",
    )
    add_enum(
        "LocalCapabilityAccess",
        "config::LocalCapabilityAccess",
        ["loopback_only", "enabled", "disabled"],
        "The immutable access policy for an arbitrary-code-execution-grade local capability.",
    )
    add_enum(
        "BatchingEffectiveSource",
        "BatchingEffectiveSource",
        [
            "default",
            "backend_policy",
            "config_file",
            "environment",
            "command_line",
            "effective_decode_width",
        ],
        "The authority that selected an effective batching value.",
    )
    add_enum(
        "DecodeBatchEffectiveSource",
        "DecodeBatchEffectiveSource",
        [
            "backend_policy",
            "config_file",
            "environment",
            "command_line",
            "deterministic",
            "max_batch_tokens",
        ],
        "The authority that selected the effective concurrent decode width.",
    )
    add_enum(
        "KtApiMode",
        "KtApiMode",
        ["auto", "all", "disabled"],
        "The process-lifetime kiln-tensor adapter route selection.",
    )
    add_enum(
        "CudaKernelProfile",
        "CudaKernelProfile",
        ["native_default", "portable_fallback"],
        "The process-lifetime CUDA backend-kernel route set.",
    )
    add_enum(
        "CudaMarlinProfile",
        "CudaMarlinProfile",
        ["disabled", "attention_mlp", "attention_mlp_gdn"],
        "The process-lifetime CUDA Marlin projection layout.",
    )
    add_enum(
        "CudaFlashBackwardMode",
        "CudaFlashBackwardMode",
        ["fast", "deterministic"],
        "The process-lifetime CUDA FlashAttention backward accumulation mode.",
    )
    add_enum(
        "MetalKernelProfile",
        "MetalKernelProfile",
        ["native_default", "portable_fallback"],
        "The process-lifetime Metal backend-kernel route set.",
    )
    add_enum(
        "RocmSynchronizationMode",
        "RocmSynchronizationMode",
        ["legacy_host_barriers", "stream_ordered"],
        "The process-lifetime ROCm synchronization policy.",
    )
    add_enum(
        "RocmStridedBatchedMatmulMode",
        "RocmStridedBatchedMatmulMode",
        ["disabled", "enabled"],
        "The process-lifetime ROCm strided-batched matmul route.",
    )
    add_enum(
        "RocmBf16MatmulOutputMode",
        "RocmBf16MatmulOutputMode",
        ["f32_then_cast", "native_bf16"],
        "The process-lifetime ROCm BF16-output matmul route.",
    )
    add_enum(
        "RocmKernelProfile",
        "RocmKernelProfile",
        ["native_default", "portable_fallback"],
        "The complete process-lifetime ROCm model-kernel route set.",
    )
    add_enum(
        "RocmGraphMode",
        "RocmGraphMode",
        ["profile", "disabled", "warmup_then_eager", "lazy_capture_replay"],
        "The configured ROCm graph lifecycle.",
    )
    add_enum(
        "RocmGraphUnavailableReason",
        "RocmGraphUnavailableReason",
        [
            "backend_without_graph_runner",
            "model_runner_busy",
            "model_runner_lock_poisoned",
            "graph_runner_busy",
            "graph_runner_lock_poisoned",
        ],
        "A closed reason why a ROCm graph snapshot could not be acquired.",
    )
    add_enum(
        "RocmGraphPhase",
        "RocmGraphPhase",
        [
            "pre_candidate_headroom",
            "candidate_warm",
            "pre_native_reservation",
            "native_capture",
            "native_replay",
            "rejected_candidate_cleanup",
        ],
        "A closed ROCm graph lifecycle or replay phase label.",
    )
    add_enum(
        "StreamingPrefillMode",
        "StreamingPrefillMode",
        ["auto", "enabled", "disabled"],
        "Configured streaming-prefill dispatch intent.",
    )
    add_enum(
        "StreamingPrefillEffectiveSource",
        "StreamingPrefillEffectiveSource",
        [
            "backend_policy",
            "default",
            "config_file",
            "environment",
            "command_line",
            "inherited_from_tile_tokens_default",
            "inherited_from_tile_tokens_config_file",
            "inherited_from_tile_tokens_environment",
            "inherited_from_tile_tokens_command_line",
        ],
        "The final authority for a resolved streaming-prefill value.",
    )
    add_enum(
        "StreamingPrefillDispatchPolicy",
        "StreamingPrefillDispatchPolicy",
        ["never", "all_non_empty", "prompt_tokens_at_least"],
        "A machine-readable streaming-prefill dispatch rule.",
    )
    add_enum(
        "SpecMethod",
        "SpecMethod",
        ["off", "skip_layer", "mtp"],
        "Configured speculative-decoding method.",
    )
    add_enum(
        "CheckpointBoundaryRecomputeMode",
        "CheckpointBoundaryRecomputeMode",
        ["auto", "enabled", "disabled"],
        "Checkpoint-boundary retention or sparse replay mode.",
    )
    add_enum(
        "ThinkingBudgetSource",
        "ThinkingBudgetSource",
        [
            "unlimited",
            "server_default",
            "request",
            "request_unlimited",
            "suite",
            "suite_unlimited",
            "run_override",
            "run_override_unlimited",
            "example",
            "example_unlimited",
            "unknown",
        ],
        "The stable origin of an effective thinking-budget value.",
    )

    add_object(
        "DeterministicInferenceDiagnostics",
        "DeterministicInferenceDiagnostics",
        {"enabled": ref("Boolean"), "source": ref("ConfigValueSource")},
        "Resolved deterministic-inference state and startup provenance.",
    )
    add_object(
        "MaxDecodeBatchDiagnostics",
        "MaxDecodeBatchDiagnostics",
        {
            "configured": nullable(ref("NonNegativeInteger")),
            "configured_source": ref("ConfigValueSource"),
            "backend_policy": ref("NonNegativeInteger"),
            "effective": ref("NonNegativeInteger"),
            "effective_source": ref("DecodeBatchEffectiveSource"),
        },
        "Configured, backend, and effective concurrent decode width.",
    )
    add_object(
        "DecodeRuntimeConfig",
        "DecodeRuntimeConfig",
        {
            "deterministic": ref("DeterministicInferenceDiagnostics"),
            "max_decode_batch": ref("MaxDecodeBatchDiagnostics"),
        },
        "The immutable process-lifetime decode policy.",
    )
    add_object(
        "BatchingToggleDiagnostics",
        "BatchingToggleDiagnostics",
        {"enabled": ref("Boolean"), "source": ref("ConfigValueSource")},
        "A resolved batching toggle and its startup provenance.",
    )
    add_object(
        "PrefillAdmissionQuantumDiagnostics",
        "PrefillAdmissionQuantumDiagnostics",
        {
            "configured": nullable(ref("NonNegativeInteger")),
            "configured_source": ref("ConfigValueSource"),
            "backend_policy": ref("NonNegativeInteger"),
            "effective": ref("NonNegativeInteger"),
            "effective_source": ref("BatchingEffectiveSource"),
        },
        "Configured, backend, and effective prefill admission quantum.",
    )
    add_object(
        "ActorCycleIdleDiagnostics",
        "ActorCycleIdleDiagnostics",
        {
            "milliseconds": ref("NonNegativeInteger"),
            "source": ref("ConfigValueSource"),
            "enabled": ref("Boolean"),
            "command_poll_milliseconds": ref("PositiveInteger"),
        },
        "Configured cooperative safe-boundary actor idle and control-command polling contract.",
    )
    add_object(
        "BatchingRuntimeConfig",
        "BatchingRuntimeConfig",
        {
            "rowwise_decode": ref("BatchingToggleDiagnostics"),
            "prefix_aware_admission": ref("BatchingToggleDiagnostics"),
            "prefill_admission_quantum": ref("PrefillAdmissionQuantumDiagnostics"),
            "actor_cycle_idle": ref("ActorCycleIdleDiagnostics"),
            "burst_prefill_admission": ref("Boolean"),
            "actor_prefill_tile_alignment_required": ref("Boolean"),
        },
        "Runtime-ready batching policy resolved once after backend selection.",
    )

    add_object(
        "ServingRuntimePolicy",
        "ServingRuntimePolicy",
        {
            "inference_admission": ref("Boolean"),
            "training_gpu_ownership": ref("Boolean"),
            "adapter_weight_transitions": ref("Boolean"),
            "dynamic_kv_resize": ref("Boolean"),
            "allocator_reclaim": ref("Boolean"),
            "live_graph_capture": ref("Boolean"),
            "exclusive_gpu_behavior": ref("String"),
        },
        "Behavior derived solely from the immutable serving profile.",
    )
    add_object(
        "ServingProfileDiagnostics",
        "ServingProfileDiagnostics",
        {
            "profile": ref("ServingProfile"),
            "source": ref("ConfigValueSource"),
            "immutable_after_startup": ref("Boolean"),
            "request_overrides_allowed": ref("Boolean"),
            "effective_policy_source": ref("String"),
            "effective_policy": ref("ServingRuntimePolicy"),
        },
        "Operator-facing resolution report for process-lifetime serving policy.",
    )
    for name, rust_value, value_schema in (
        ("ResolvedKtApiMode", "KtApiMode", ref("KtApiMode")),
        (
            "ResolvedCudaKernelProfile",
            "CudaKernelProfile",
            ref("CudaKernelProfile"),
        ),
        (
            "ResolvedCudaMarlinProfile",
            "CudaMarlinProfile",
            ref("CudaMarlinProfile"),
        ),
        (
            "ResolvedCudaFlashBackwardMode",
            "CudaFlashBackwardMode",
            ref("CudaFlashBackwardMode"),
        ),
        (
            "ResolvedMetalKernelProfile",
            "MetalKernelProfile",
            ref("MetalKernelProfile"),
        ),
        ("ResolvedRocmSynchronizationMode", "RocmSynchronizationMode", ref("RocmSynchronizationMode")),
        (
            "ResolvedRocmStridedBatchedMatmulMode",
            "RocmStridedBatchedMatmulMode",
            ref("RocmStridedBatchedMatmulMode"),
        ),
        (
            "ResolvedRocmBf16MatmulOutputMode",
            "RocmBf16MatmulOutputMode",
            ref("RocmBf16MatmulOutputMode"),
        ),
        (
            "ResolvedRocmKernelProfile",
            "RocmKernelProfile",
            ref("RocmKernelProfile"),
        ),
        ("ResolvedRocmGraphMode", "RocmGraphMode", ref("RocmGraphMode")),
        ("ResolvedAcceleratorInteger", "usize | u64", ref("NonNegativeInteger")),
        ("ResolvedAcceleratorOptionalInteger", "Option<usize>", nullable(ref("NonNegativeInteger"))),
        ("ResolvedAcceleratorBoolean", "bool", ref("Boolean")),
    ):
        add_object(
            name,
            f"ResolvedAcceleratorValue<{rust_value}>",
            {
                "configured": value_schema,
                "effective": value_schema,
                "source": ref("ConfigValueSource"),
            },
            "One configured/effective accelerator policy leaf and its startup source.",
        )
    add_object(
        "ResolvedAcceleratorRuntimePolicy",
        "ResolvedAcceleratorRuntimePolicy",
        {
            "schema_id": ref("NonEmptyString"),
            "version": ref("NonNegativeInteger"),
            "vulkan_kernel_policy_schema_id": ref("NonEmptyString"),
            "vulkan_device_policy_schema_id": ref("NonEmptyString"),
            "serving_profile": ref("ServingProfile"),
            "serving_profile_source": ref("ConfigValueSource"),
            "kt_api_mode": ref("ResolvedKtApiMode"),
            "full_attention_score_budget_mib": ref("ResolvedAcceleratorInteger"),
            "vulkan_device_index": ref("ResolvedAcceleratorOptionalInteger"),
            "vulkan_validation": ref("ResolvedAcceleratorBoolean"),
            "cuda_kernel_profile": ref("ResolvedCudaKernelProfile"),
            "cuda_marlin_profile": ref("ResolvedCudaMarlinProfile"),
            "cuda_flash_backward_mode": ref("ResolvedCudaFlashBackwardMode"),
            "metal_kernel_profile": ref("ResolvedMetalKernelProfile"),
            "rocm_synchronization_mode": ref("ResolvedRocmSynchronizationMode"),
            "rocm_strided_batched_matmul_mode": ref(
                "ResolvedRocmStridedBatchedMatmulMode"
            ),
            "rocm_bf16_matmul_output_mode": ref("ResolvedRocmBf16MatmulOutputMode"),
            "rocm_kernel_profile": ref("ResolvedRocmKernelProfile"),
            "rocm_graph_mode": ref("ResolvedRocmGraphMode"),
            "rocm_graph_cache_entries": ref("ResolvedAcceleratorInteger"),
            "rocm_graph_cache_max_bytes": ref("ResolvedAcceleratorInteger"),
        },
        "Versioned process-lifetime accelerator policy.",
    )

    add_object(
        "StreamingPrefillDispatchRuleDiagnostics",
        "StreamingPrefillDispatchRuleDiagnostics",
        {
            "policy": ref("StreamingPrefillDispatchPolicy"),
            "minimum_prompt_tokens": ref("NonNegativeInteger"),
        },
        "A stable machine-readable streaming-prefill dispatch rule.",
        optional=("minimum_prompt_tokens",),
    )
    add_object(
        "StreamingPrefillDispatchDiagnostics",
        "StreamingPrefillDispatchDiagnostics",
        {
            "configured_mode": ref("StreamingPrefillMode"),
            "configured_source": ref("ConfigValueSource"),
            "backend_policy": ref("StreamingPrefillDispatchRuleDiagnostics"),
            "effective": ref("StreamingPrefillDispatchRuleDiagnostics"),
            "effective_source": ref("StreamingPrefillEffectiveSource"),
        },
        "Configured, backend, and effective streaming-prefill dispatch.",
    )
    add_object(
        "StreamingPrefillThresholdDiagnostics",
        "StreamingPrefillThresholdDiagnostics",
        {
            "configured": nullable(ref("NonNegativeInteger")),
            "configured_source": ref("ConfigValueSource"),
            "backend_policy": nullable(ref("NonNegativeInteger")),
            "effective_for_auto_mode": nullable(ref("NonNegativeInteger")),
            "override_applied_to_backend_auto_policy": ref("Boolean"),
        },
        "Configured and backend streaming-prefill threshold resolution.",
    )
    add_object(
        "StreamingPrefillTileDiagnostics",
        "StreamingPrefillTileDiagnostics",
        {
            "configured": nullable(ref("NonNegativeInteger")),
            "configured_source": ref("ConfigValueSource"),
            "backend_policy": ref("NonNegativeInteger"),
            "effective": ref("NonNegativeInteger"),
            "effective_source": ref("StreamingPrefillEffectiveSource"),
        },
        "Configured, backend, and effective streaming-prefill tile size.",
    )
    add_object(
        "StreamingPrefillDerivedTileDiagnostics",
        "StreamingPrefillDerivedTileDiagnostics",
        {
            "backend_policy": ref("NonNegativeInteger"),
            "effective": ref("NonNegativeInteger"),
            "effective_source": ref("StreamingPrefillEffectiveSource"),
        },
        "Backend and effective derived streaming-prefill tile size.",
    )
    add_object(
        "StreamingPrefillToggleDiagnostics",
        "StreamingPrefillToggleDiagnostics",
        {
            "configured": ref("Boolean"),
            "configured_source": ref("ConfigValueSource"),
            "effective": ref("Boolean"),
            "effective_source": ref("StreamingPrefillEffectiveSource"),
        },
        "Configured and effective streaming-prefill boolean policy.",
    )
    add_object(
        "StreamingPrefillRuntimeConfig",
        "StreamingPrefillRuntimeConfig",
        {
            "dispatch": ref("StreamingPrefillDispatchDiagnostics"),
            "threshold_tokens": ref("StreamingPrefillThresholdDiagnostics"),
            "tile_tokens": ref("StreamingPrefillTileDiagnostics"),
            "tape_tile_tokens": ref("StreamingPrefillTileDiagnostics"),
            "detached_full_attn_tile_tokens": ref("StreamingPrefillTileDiagnostics"),
            "detached_full_attn_boundary_tile_tokens": ref("StreamingPrefillDerivedTileDiagnostics"),
            "detached_full_attn_tape_replay_tile_tokens": ref("StreamingPrefillDerivedTileDiagnostics"),
            "last_token_lm_head": ref("StreamingPrefillToggleDiagnostics"),
            "immutable_after_startup": ref("Boolean"),
            "restart_required_to_change": ref("Boolean"),
        },
        "Complete operator-facing resolved streaming-prefill policy.",
    )

    add_enum(
        "CudaSynchronizationReason",
        "CudaSyncReason",
        [
            "explicit_device_drain",
            "explicit_stream_drain",
            "tensor_handoff",
            "external_yield",
            "in_place_mutation",
            "memory_reclaim",
            "graph_boundary",
            "full_attention_handoff",
            "model_handoff",
            "host_readback",
            "allocation_lifetime",
            "global_state_mutation",
        ],
        "Closed reason vocabulary for CUDA host synchronization.",
    )
    add_object(
        "CudaSynchronizationReasonStats",
        "CudaSynchronizationReasonStats",
        {
            "reason": ref("CudaSynchronizationReason"),
            "device_wait_count": ref("NonNegativeInteger"),
            "stream_wait_count": ref("NonNegativeInteger"),
            "failure_count": ref("NonNegativeInteger"),
            "waited_ns": ref("NonNegativeInteger"),
        },
        "Fixed-cardinality counters for one CUDA synchronization reason.",
    )
    add_object(
        "CudaSynchronizationRuntimeStats",
        "CudaSynchronizationRuntimeStats",
        {
            "active": ref("Boolean"),
            "telemetry_available": ref("Boolean"),
            "telemetry_error": ref("String"),
            "total_device_wait_count": ref("NonNegativeInteger"),
            "total_stream_wait_count": ref("NonNegativeInteger"),
            "total_failure_count": ref("NonNegativeInteger"),
            "total_waited_ns": ref("NonNegativeInteger"),
            "reasons": {
                "oneOf": [
                    array(ref("CudaSynchronizationReasonStats"), min_items=0, max_items=0),
                    array(ref("CudaSynchronizationReasonStats"), min_items=12, max_items=12),
                ]
            },
        },
        "Point-in-time CUDA synchronization telemetry.",
        optional=("telemetry_error",),
    )

    add_object(
        "RocmSynchronizationReasonStats",
        "RocmSynchronizationReasonStats",
        {
            "reason": ref("NonEmptyString"),
            "device_wait_count": ref("NonNegativeInteger"),
            "stream_wait_count": ref("NonNegativeInteger"),
            "waited_ns": ref("NonNegativeInteger"),
            "skipped_count": ref("NonNegativeInteger"),
        },
        "Fixed-cardinality counters for one ROCm synchronization reason.",
    )
    add_object(
        "RocmSynchronizationRuntimeStats",
        "RocmSynchronizationRuntimeStats",
        {
            "active": ref("Boolean"),
            "telemetry_available": ref("Boolean"),
            "cleanup_quarantined": ref("Boolean"),
            "telemetry_error": ref("String"),
            "total_device_wait_count": ref("NonNegativeInteger"),
            "total_stream_wait_count": ref("NonNegativeInteger"),
            "total_waited_ns": ref("NonNegativeInteger"),
            "total_skipped_count": ref("NonNegativeInteger"),
            "reasons": array(ref("RocmSynchronizationReasonStats")),
        },
        "Point-in-time ROCm synchronization telemetry.",
        optional=("telemetry_error",),
    )
    add_object(
        "ExternalYieldSyncStats",
        "ExternalYieldSyncStats",
        {
            "boundary": ref("String"),
            "calls": ref("NonNegativeInteger"),
            "failures": ref("NonNegativeInteger"),
            "total_micros": ref("NonNegativeInteger"),
            "max_micros": ref("NonNegativeInteger"),
            "slow_calls": ref("NonNegativeInteger"),
        },
        "Process-lifetime settlement counters at one external-yield boundary.",
    )
    add_object(
        "RocmGraphPhaseStats",
        "RocmGraphPhaseStats",
        {
            "calls": ref("NonNegativeInteger"),
            "slow": ref("NonNegativeInteger"),
            "total_duration_micros": ref("NonNegativeInteger"),
            "max_duration_micros": ref("NonNegativeInteger"),
        },
        "Bounded latency telemetry for one ROCm graph lifecycle or replay phase.",
    )
    fallback_fields = [
        "total",
        "multi_row_batch_unsupported",
        "cold_cache_host_round_trip",
        "persistent_host_round_trip",
        "shape_dependent_attention",
        "graph_cache_capacity",
        "graph_cache_byte_budget",
        "graph_accounting_incomplete",
        "moderate_memory_pressure",
        "tight_memory_pressure",
        "critical_memory_pressure",
        "memory_reservation_denied",
        "memory_governor_selector_mismatch",
        "capture_failure",
        "replay_failure",
        "slow",
        "total_duration_micros",
        "max_duration_micros",
    ]
    add_object(
        "RocmGraphFallbackStats",
        "RocmGraphFallbackStats",
        {field: ref("NonNegativeInteger") for field in fallback_fields},
        "Closed-reason ROCm eager-fallback counts and end-to-end latency.",
    )
    add_object(
        "RocmGraphLiveTelemetry",
        "RocmGraphLiveTelemetry",
        {
            "current_phase": nullable(ref("RocmGraphPhase")),
            "current_phase_elapsed_micros": ref("NonNegativeInteger"),
            "pre_candidate_headroom_phase": ref("RocmGraphPhaseStats"),
            "candidate_warm_phase": ref("RocmGraphPhaseStats"),
            "pre_native_reservation_phase": ref("RocmGraphPhaseStats"),
            "native_capture_phase": ref("RocmGraphPhaseStats"),
            "native_replay_phase": ref("RocmGraphPhaseStats"),
            "rejected_candidate_cleanup_phase": ref("RocmGraphPhaseStats"),
            "last_transient_candidate_bytes": ref("NonNegativeInteger"),
            "peak_transient_candidate_bytes": ref("NonNegativeInteger"),
        },
        "Graph-runner-lock-independent ROCm graph phase telemetry.",
    )
    rocm_bool_fields = ["requested", "capture_requested", "enabled", "capture_enabled", "retained_bytes_accounting_complete"]
    rocm_phase_fields = [
        "pre_candidate_headroom_phase",
        "candidate_warm_phase",
        "pre_native_reservation_phase",
        "native_capture_phase",
        "native_replay_phase",
        "rejected_candidate_cleanup_phase",
    ]
    rocm_integer_fields = [
        "max_cached_graphs", "max_retained_bytes", "capture_attempts", "capture_successes",
        "capture_deferrals", "capture_failures", "replay_attempts", "replay_successes",
        "replay_failures", "failures", "decode_owner_release_count", "decode_owner_graph_release_count",
        "graph_slot_create_count", "graph_slot_reuse_count", "cache_admission_successes", "cache_evictions",
        "cache_evicted_bytes", "budget_evictions", "pressure_evictions", "invalidation_evictions",
        "recovery_evictions", "entry_capacity_rejections", "byte_budget_rejections",
        "accounting_incomplete_rejections", "pre_capture_entry_capacity_skips",
        "pre_capture_byte_budget_skips", "pre_capture_accounting_incomplete_skips",
        "pre_capture_memory_reservation_denied_skips", "memory_governor_selector_mismatch_skips",
        "captured_graph_count", "graph_slot_count", "active_graph_slot_count", "idle_graph_slot_count",
        "tracked_decode_owner_count", "retained_stable_io_bytes", "retained_capture_arena_bytes",
        "retained_blaslt_workspace_bytes", "retained_slot_state_bytes", "retained_bytes",
        "peak_retained_bytes", "opaque_native_object_count", "quarantined_retained_bytes",
        "last_transient_candidate_bytes", "peak_transient_candidate_bytes",
    ]
    rocm_stats_fields = {field: ref("Boolean") for field in rocm_bool_fields}
    rocm_stats_fields.update({field: ref("NonNegativeInteger") for field in rocm_integer_fields})
    rocm_stats_fields.update({field: ref("RocmGraphPhaseStats") for field in rocm_phase_fields})
    rocm_stats_fields["fallbacks"] = ref("RocmGraphFallbackStats")
    add_object(
        "RocmGraphStats",
        "RocmGraphStats",
        rocm_stats_fields,
        "Point-in-time ROCm HIP-graph execution, ownership, memory, and fallback state.",
    )

    add_object(
        "KvAutoscalerState",
        "KvAutoscalerState",
        {
            "requested": ref("Boolean"),
            "requested_source": ref("ConfigValueSource"),
            "force_blocks": nullable(ref("NonNegativeInteger")),
            "force_blocks_source": ref("ConfigValueSource"),
            "enabled": ref("Boolean"),
            "state": ref("String"),
            "reason": ref("String"),
            "start_blocks": nullable(ref("NonNegativeInteger")),
            "min_blocks": nullable(ref("NonNegativeInteger")),
            "bytes_per_block": nullable(ref("NonNegativeInteger")),
        },
        "Startup and activation state for the physical KV-cache autoscaler.",
    )
    add_object(
        "ModelDefaultsProfile",
        "ModelDefaultsProfile",
        {
            "name": ref("NonEmptyString"),
            "canonical_model_id": ref("NonEmptyString"),
            "canonical_served_model_id": ref("NonEmptyString"),
            "server_default_thinking_enabled": nullable(ref("Boolean")),
            "template_default_thinking_enabled": ref("Boolean"),
            "eval_mode_default_thinking_enabled": ref("Boolean"),
            "adapter_dir_policy": ref("String"),
            "chat_template_policy": ref("String"),
            "supports_enable_thinking_kwarg": ref("Boolean"),
            "supports_tool_chat_template": ref("Boolean"),
        },
        "Built-in runtime defaults for the supported model family.",
    )
    add_object(
        "ConfigHashes",
        "ConfigHashes",
        {
            "tokenizer_config_hash": ref("Sha256"),
            "chat_template_hash": ref("Sha256"),
            "training_chat_template_hash": ref("Sha256"),
            "model_config_hash": ref("Sha256"),
            "effective_config_hash": ref("Sha256"),
        },
        "Available hashes of loaded model, tokenizer, template, and effective environment configuration.",
        optional=(
            "tokenizer_config_hash", "chat_template_hash", "training_chat_template_hash",
            "model_config_hash", "effective_config_hash",
        ),
    )

    add_object(
        "CheckpointBoundaryPolicy",
        "CheckpointBoundaryPolicy",
        {
            "recompute_mode": ref("CheckpointBoundaryRecomputeMode"),
            "recompute_threshold_tokens": ref("PositiveInteger"),
            "anchor_stride": nullable(ref("PositiveInteger")),
            "cache_target_bytes": ref("PositiveInteger"),
        },
        "Resolved checkpoint-boundary retention and sparse replay policy.",
    )
    add_definition(
        "GradientCheckpointPolicy",
        "GradientCheckpointPolicy",
        {
            "oneOf": [
                {
                    "type": "object", "additionalProperties": False, "required": ["mode"],
                    "properties": {"mode": {"const": "auto"}},
                },
                {
                    "type": "object", "additionalProperties": False, "required": ["mode", "segments"],
                    "properties": {"mode": {"const": "explicit_segments"}, "segments": ref("PositiveInteger")},
                },
                {
                    "type": "object", "additionalProperties": False, "required": ["mode", "segments"],
                    "properties": {"mode": {"const": "disabled"}, "segments": nullable(ref("PositiveInteger"))},
                },
            ]
        },
        "Tagged immutable gradient-checkpoint behavior for one native training run.",
    )

    add_object(
        "BaseWeightShardIdentity",
        "BaseWeightShardIdentity",
        {
            "filename": described(ref("NonEmptyString"), "Path-safe logical safetensors shard filename."),
            "size_bytes": ref("PositiveInteger"),
            "sha256": ref("Sha256"),
        },
        "Exact identity of one logical safetensors shard.",
    )
    add_object(
        "BaseWeightShardManifest",
        "BaseWeightShardManifest",
        {
            "schema_version": {"const": 1},
            "manifest_type": {"const": "kiln.base-weight-shards.v1"},
            "aggregate_algorithm": {"const": "kiln.base-model-content.v1"},
            "aggregate_sha256": ref("Sha256"),
            "total_size_bytes": ref("PositiveInteger"),
            "shards": array(ref("BaseWeightShardIdentity"), min_items=1),
        },
        "Canonical portable manifest for the exact loaded base-weight shards.",
    )
    add_object(
        "ExecutionBackendIdentity",
        "ExecutionBackendIdentity",
        {"name": ref("NonEmptyString"), "device": ref("NonEmptyString"), "numerical_runtime_sha256": ref("Sha256")},
        "Backend and numerical-runtime identity.",
    )
    add_object(
        "ExecutionBuildIdentity",
        "ExecutionBuildIdentity",
        {
            "package_version": ref("NonEmptyString"), "target": ref("NonEmptyString"),
            "executable_sha256": ref("Sha256"), "git_commit": ref("NonEmptyString"),
            "source_tree_sha256": ref("Sha256"), "source_dirty": ref("Boolean"),
        },
        "Executable build identity.",
        optional=("git_commit", "source_tree_sha256", "source_dirty"),
    )
    add_object(
        "ExecutionModelIdentity",
        "ExecutionModelIdentity",
        {
            "model_config_sha256": ref("Sha256"), "tokenizer_vocab_sha256": ref("Sha256"),
            "tokenizer_config_sha256": ref("Sha256"), "chat_template_sha256": ref("Sha256"),
            "training_chat_template_sha256": ref("Sha256"),
        },
        "Loaded model, tokenizer, and template identity.",
        optional=("chat_template_sha256", "training_chat_template_sha256"),
    )
    add_object(
        "ExecutionPrecisionIdentity",
        "ExecutionPrecisionIdentity",
        {"inference_dtype": ref("NonEmptyString"), "training_policy": ref("NonEmptyString")},
        "Resolved inference and training precision identity.",
    )
    add_object(
        "ExecutionKernelIdentity",
        "ExecutionKernelIdentity",
        {
            "contract_type": {"const": "kiln.kernel-contract.v1"},
            "versions": mapping(ref("NonEmptyString")),
            "compiled_features": array(ref("NonEmptyString")),
            "contract_sha256": ref("Sha256"),
        },
        "Versioned compiled-kernel contract identity.",
    )
    add_object(
        "ExecutionConfigurationIdentity",
        "ExecutionConfigurationIdentity",
        {"effective_server_config_sha256": ref("Sha256"), "effective_environment_sha256": ref("Sha256")},
        "Effective server configuration and environment identity.",
    )
    add_object(
        "ExecutionProvenanceV1",
        "ExecutionProvenanceV1",
        {
            "schema_version": {"const": 1},
            "provenance_type": {"const": "kiln.execution-provenance.v1"},
            "backend": ref("ExecutionBackendIdentity"), "build": ref("ExecutionBuildIdentity"),
            "model": ref("ExecutionModelIdentity"), "precision": ref("ExecutionPrecisionIdentity"),
            "kernels": ref("ExecutionKernelIdentity"), "configuration": ref("ExecutionConfigurationIdentity"),
            "provenance_sha256": ref("Sha256"),
        },
        "Immutable self-verifying execution envelope captured at server startup.",
    )

    add_object(
        "RocmGraphInfo",
        "RocmGraphInfo",
        {
            **{field: nullable(ref("Boolean")) for field in rocm_bool_fields[:4]},
            "state": {"type": "string", "enum": ["enabled", "disabled", "busy", "unavailable"]},
            "unavailable_reason": nullable(ref("RocmGraphUnavailableReason")),
            "phase_telemetry_available": ref("Boolean"),
            "phase_telemetry_unavailable_reason": nullable(ref("RocmGraphUnavailableReason")),
            "current_phase": nullable(ref("RocmGraphPhase")),
            "current_phase_elapsed_micros": nullable(ref("NonNegativeInteger")),
            **{field: nullable(ref("NonNegativeInteger")) for field in rocm_integer_fields},
            "retained_bytes_accounting_complete": nullable(ref("Boolean")),
            **{field: nullable(ref("RocmGraphPhaseStats")) for field in rocm_phase_fields},
            "fallbacks": nullable(ref("RocmGraphFallbackStats")),
        },
        "Nonblocking ROCm graph state projected for health reporting.",
    )
    add_object(
        "GraphInfo",
        "GraphInfo",
        {"enabled": nullable(ref("Boolean")), "state": {"type": "string", "enum": ["enabled", "disabled", "busy"]}},
        "Nonblocking CUDA or Metal graph state.",
    )
    add_object(
        "CudaGraphInfo",
        "CudaGraphInfo",
        {
            "requested": ref("Boolean"),
            "capture_allowed_by_serving_profile": ref("Boolean"),
            "enabled": nullable(ref("Boolean")),
            "state": {"type": "string", "enum": ["enabled", "disabled", "busy"]},
            "max_cached_graphs": ref("PositiveInteger"),
            "stable_paged_metadata": {"const": True},
            "batched_capture_available": {"const": False},
            "restart_required_to_change": {"const": True},
        },
        "Configured CUDA graph request, live runner state, bounded cache, and fixed safety invariants.",
    )
    batch_snapshot_integer_fields = [
        "snapshot_age_ms", "stream_stall_grace_ms", "actor_cycle_idle_ms", "actor_cycle_idle_count",
        "queue_depth", "active_decode", "active_prefill",
        "active_resident_prefill",
        "max_batch_tokens", "max_prefill_tokens_per_cycle", "max_prefill_layers_per_cycle",
        "max_prefill_admission_quantum", "max_prefill_staging_slots", "max_active_requests",
        "max_prefill_staging_priority_burst", "max_decode_batch", "active_staged_requests",
        "max_observed_active_requests", "current_batch_size", "last_batch_size", "max_observed_batch_size",
        "slow_decode_forward_count", "slow_prefill_forward_count", "last_prefill_tokens", "last_prefill_layers",
        "total_admission_calls", "slow_admission_count", "total_decode_forwards", "total_batched_decode_forwards",
        "total_decode_rows", "total_prefill_admission_cycles", "total_prefill_forwards",
        "total_resident_prefill_attempts", "total_resident_prefill_forwards",
        "total_resident_prefill_initial_declines", "total_resident_prefill_route_failures",
        "total_resident_prefill_rows", "total_resident_prefill_completed_rows",
        "last_resident_prefill_batch_size", "max_resident_prefill_batch_size", "total_decode_tokens",
        "total_prefill_tokens", "total_prefill_layers", "total_prefill_layer_yields",
        "total_short_prefill_priority_forwards", "total_prefill_staging_priority_forwards",
        "total_prefill_staging_admissions", "total_errors", "response_delivery_in_flight",
        "response_delivery_backpressured", "response_delivery_pending_terminal", "response_backpressure_events",
        "response_backpressure_wait_ms", "response_stall_evictions", "response_channel_closed",
        "adapter_groups_waiting", "prefix_deferred_waiting", "prefix_admission_deferrals",
    ]
    batch_snapshot_number_fields = [
        "last_forward_ms", "max_decode_forward_ms", "total_decode_forward_ms", "last_prefill_ms",
        "max_prefill_forward_ms", "total_prefill_forward_ms", "last_admission_ms", "max_admission_ms",
        "total_admission_ms", "total_actor_cycle_idle_ms", "max_actor_cycle_idle_ms",
    ]
    batch_snapshot_fields = {field: ref("NonNegativeInteger") for field in batch_snapshot_integer_fields}
    batch_snapshot_fields.update({field: ref("NonNegativeNumber") for field in batch_snapshot_number_fields})
    batch_snapshot_fields["stream_stall_grace_source"] = ref("ConfigValueSource")
    batch_snapshot_fields["actor_cycle_idle_source"] = ref("ConfigValueSource")
    batch_snapshot_fields["actor_cycle_idle_active"] = ref("Boolean")
    batch_snapshot_fields["actor_barrier_adapter_active"] = ref("Boolean")
    batch_snapshot_fields["actor_barrier_resize_active"] = ref("Boolean")
    batch_snapshot_fields["accepting"] = ref("Boolean")
    batch_snapshot_fields["max_batch_tokens_source"] = ref("ConfigValueSource")
    batch_snapshot_fields["max_prefill_tokens_per_cycle_source"] = ref("ConfigValueSource")
    batch_snapshot_fields["max_prefill_layers_per_cycle_source"] = ref("ConfigValueSource")
    add_object(
        "BatchingEngineSnapshot",
        "BatchingEngineInfo | BatchingEngineSnapshotDebug",
        batch_snapshot_fields,
        "Cached batching-engine admission, execution, latency, backpressure, and fairness counters.",
    )
    add_object(
        "MemoryGovernorRuntimeInfo",
        "MemoryGovernorRuntimeInfo",
        {
            **{
                field: ref("NonNegativeInteger")
                for field in [
                    "sample_age_ms", "sample_max_age_ms", "automatic_attempts",
                    "automatic_successful_attempts", "automatic_zero_yield_attempts",
                    "automatic_suppressed_attempts", "automatic_reclaimed_bytes",
                    "automatic_last_target_bytes", "automatic_last_reclaimed_bytes",
                    "automatic_last_duration_us", "automatic_retry_after_ms", "automatic_zero_yield_streak",
                ]
            },
            "reclaim_mode": ref("String"), "requested_reclaim_mode": ref("String"),
            "automatic_monitor_enabled": ref("Boolean"), "sampler_required": ref("Boolean"),
            "sampler_running": ref("Boolean"), "sampler_healthy": ref("Boolean"),
            "sample_stale": ref("Boolean"), "source": ref("ConfigValueSource"),
            "disabled_by_serving_profile": ref("Boolean"),
        },
        "Live memory-governor mode, sampler health, and automatic reclaim counters.",
    )
    add_object(
        "DecodeRuntimeInfo",
        "DecodeRuntimeInfo",
        {
            "configuration": ref("DecodeRuntimeConfig"),
            "accelerator_runtime": ref("ResolvedAcceleratorRuntimePolicy"),
            "cuda_synchronization": ref("CudaSynchronizationRuntimeStats"),
            "rocm_synchronization": ref("RocmSynchronizationRuntimeStats"),
            "batching_configuration": ref("BatchingRuntimeConfig"),
            "cuda_graphs": ref("CudaGraphInfo"), "rocm_graphs": ref("RocmGraphInfo"), "metal_graphs": ref("GraphInfo"),
            "kv_autoscaler": ref("KvAutoscalerState"), "memory_governor": ref("MemoryGovernorRuntimeInfo"),
            "batching_engine": nullable(ref("BatchingEngineSnapshot")),
        },
        "Resolved decode configuration and live backend execution state.",
    )

    add_object("HttpRuntimeInfo", "HttpRuntimeInfo | HttpDebugState", {
        "send_buffer_requested_bytes": nullable(ref("NonNegativeInteger")),
        "send_buffer_kernel_readback_bytes": nullable(ref("NonNegativeInteger")),
        "send_buffer_effective_bytes": nullable(ref("NonNegativeInteger")),
    }, "Resolved and kernel-readback HTTP socket send-buffer state.")
    add_object("BackendRuntimeInfo", "BackendRuntimeInfo | BackendRuntimeDebugState", {
        "healthy": ref("Boolean"), "quarantined": ref("Boolean"), "reason": nullable(ref("String")),
        "restart_required": ref("Boolean"), "external_yield_sync": array(ref("ExternalYieldSyncStats")),
    }, "Backend health, quarantine, restart, and external-yield settlement state.")
    add_object("BaseWeightIdentitySummary", "BaseWeightIdentitySummary", {
        "manifest_type": ref("String"), "aggregate_algorithm": ref("String"), "aggregate_sha256": ref("Sha256"),
        "shard_count": ref("NonNegativeInteger"), "total_size_bytes": ref("NonNegativeInteger"),
    }, "Compact loaded base-weight identity.")
    add_object("ExecutionIdentitySummary", "ExecutionIdentitySummary", {
        field: ref("String") for field in [
            "provenance_type", "backend", "device", "inference_dtype", "training_policy"
        ]
    } | {
        field: ref("Sha256") for field in [
            "provenance_sha256", "executable_sha256", "numerical_runtime_sha256", "kernel_contract_sha256",
            "effective_server_config_sha256", "effective_environment_sha256",
        ]
    }, "Compact execution-provenance identity.")
    add_object("SchedulerStats", "SchedulerStats", {
        field: ref("NonNegativeInteger") for field in ["waiting", "running", "blocks_used", "blocks_free", "blocks_total"]
    }, "Scheduler and block-manager gauges.")
    add_object("LiveMemoryTier", "LiveMemoryTier", {
        field: ref("NonNegativeNumber") for field in ["total_gb", "used_gb", "free_gb"]
    }, "One independently constrained live memory tier.")
    add_object("LiveMemory", "LiveMemory", {
        "probe_failed": ref("Boolean"),
        **{field: ref("NonNegativeNumber") for field in ["total_gb", "used_gb", "free_gb", "available_gb", "soft_reserved_gb"]},
        "pressure": ref("String"), "source": ref("String"), "unified": ref("Boolean"),
        "sample_age_ms": ref("NonNegativeInteger"), "sample_max_age_ms": ref("NonNegativeInteger"),
        "sample_stale": ref("Boolean"), "sampler_required": ref("Boolean"),
        "sampler_running": ref("Boolean"), "sampler_healthy": ref("Boolean"),
        "host_backed": nullable(ref("LiveMemoryTier")),
    }, "The memory governor's current all-process memory observation.")
    add_object("GpuMemoryInfo", "GpuMemoryInfo", {
        **{field: ref("NonNegativeInteger") for field in [
            "total_vram_bytes", "model_bytes", "estimated_model_bytes", "post_load_used_bytes",
            "peak_prefill_used_bytes", "kv_cache_bytes", "training_budget_bytes", "allocated_bytes", "reserved_bytes",
        ]},
        **{field: ref("NonNegativeNumber") for field in [
            "total_vram_gb", "model_gb", "estimated_model_gb", "post_load_used_gb", "peak_prefill_used_gb",
            "kv_cache_gb", "training_budget_gb", "allocated_gb", "reserved_gb", "inference_memory_fraction",
        ]},
        "live": nullable(ref("LiveMemory")),
    }, "Startup GPU memory budget plus the live all-process memory observation.")
    add_object("VulkanBufferInfo", "VulkanBufferInfo", {
        field: ref("NonNegativeInteger") for field in [
            "live_device_local_buffers", "live_device_local_bytes",
            "live_host_visible_buffers", "live_host_visible_bytes", "peak_live_bytes",
            "device_local_allocations", "device_local_allocated_bytes",
            "device_local_frees", "device_local_freed_bytes",
            "host_visible_allocations", "host_visible_allocated_bytes",
            "host_visible_frees", "host_visible_freed_bytes",
        ]
    }, "Process-lifetime live and cumulative VulkanBuffer allocation accounting by memory route.")
    add_object("VulkanBufferPoolInfo", "VulkanBufferPoolInfo", {
        field: ref("NonNegativeInteger") for field in [
            "max_retained_bytes", "bucket_count", "buffer_count", "retained_bytes",
            "free_buffer_count", "free_bytes", "borrowed_buffer_count", "borrowed_bytes",
            "cache_hits", "cache_misses", "eviction_count", "evicted_bytes",
            "uncached_allocation_count", "uncached_allocated_bytes",
        ]
    }, "Bounded Vulkan scratch-recycler ownership, effectiveness, eviction, and overflow accounting.")
    add_object("RequestMetrics", "RequestMetrics", {
        field: ref("NonNegativeInteger") for field in ["total", "ok", "error", "timeout", "rejected", "active", "active_peak"]
    }, "Lifetime request outcome and concurrency counters.")
    add_object("LatencyPercentiles", "LatencyPercentiles", {
        field: ref("NonNegativeNumber") for field in ["p50", "p95", "p99"]
    }, "Latency percentiles in milliseconds.")
    add_object("LastErrorSummary", "LastErrorSummary", {
        "id": ref("String"), "timestamp_unix_ms": ref("NonNegativeInteger"), "finish_reason": ref("String"),
        "error": nullable(ref("String")), "duration_ms": ref("NonNegativeInteger"), "adapter": nullable(ref("String")),
    }, "Compact summary of the most recent failed or timed-out request.")
    add_object("RecentRequestMetrics", "RecentRequestMetrics", {
        "retained": ref("NonNegativeInteger"), "capacity": ref("NonNegativeInteger"),
        "latency_ms": ref("LatencyPercentiles"), "tokens_per_second": ref("NonNegativeNumber"),
        "timeout_count": ref("NonNegativeInteger"), "error_count": ref("NonNegativeInteger"),
        "last_error": nullable(ref("LastErrorSummary")),
    }, "Aggregated metrics over the bounded recent-request ring.")
    add_object("PrefixCacheInfo", "PrefixCacheInfo | PrefixCacheDebugState", {
        "enabled": ref("Boolean"),
        **{field: ref("NonNegativeInteger") for field in [
            "lookup_hits", "lookup_misses", "hit_tokens", "hit_blocks", "cached_blocks", "max_blocks",
            "cached_entries", "max_entries", "cached_state_bytes", "max_state_bytes", "active_leases",
            "pending_release_entries",
        ]},
    }, "Prefix-cache occupancy, hit, limit, and lease counters.")
    add_object("PrefixCacheHealthInfo", "PrefixCacheInfo", {
        **DEFS["PrefixCacheInfo"]["properties"],
        "block_utilization": ref("NonNegativeNumber"), "entry_utilization": ref("NonNegativeNumber"),
        "state_utilization": ref("NonNegativeNumber"),
    }, "Prefix-cache counters plus normalized utilization for health reporting.")
    add_object("PromptCacheInfo", "PromptCacheInfo | PromptCacheDebugState", {
        "hits": ref("NonNegativeInteger"), "misses": ref("NonNegativeInteger"), "entries": ref("NonNegativeInteger"),
    }, "One bounded prompt-cache hit and occupancy snapshot.")
    add_object("PromptCachesInfo", "PromptCachesInfo", {
        "rendered_prompt": ref("PromptCacheInfo"), "prompt_token": ref("PromptCacheInfo"),
    }, "Rendered-prompt and prompt-token cache snapshots.")
    add_object("PrefillRuntimeInfo", "PrefillRuntimeInfo", {
        "streaming_prefill": ref("StreamingPrefillRuntimeConfig"),
    }, "Resolved prefill execution state.")
    add_object("ActiveJobInfo", "ActiveJobInfo", {
        "job_id": ref("String"), "progress": ref("FiniteNumber"),
    }, "The active training job and its reported progress.")
    add_object("TrainingInfo", "TrainingInfo", {
        "active_job": nullable(ref("ActiveJobInfo")), "queued": ref("NonNegativeInteger"),
        "checkpoint_boundary_policy": ref("CheckpointBoundaryPolicy"),
    }, "Live training queue and checkpoint-boundary state.")
    add_object("HealthCheck", "HealthCheck", {
        "name": {
            "type": "string",
            "enum": [
                "model_loaded",
                "scheduler_responsive",
                "backend_runtime_healthy",
                "inference_admission",
                "inference_prewarm_complete",
                "execution_provenance_valid",
            ],
        },
        "pass": ref("Boolean"),
    }, "One stable readiness condition and its result.")
    add_object("SelfImproveSchedulerStatus", "SelfImproveSchedulerStatus", {
        "interval_hours": ref("NonNegativeInteger"), "last_run_unix_ms": ref("NonNegativeInteger"),
        "last_result": ref("String"), "last_job_ids": array(ref("String")),
        "next_run_unix_ms": ref("NonNegativeInteger"),
    }, "Durable self-improvement scheduler status.", optional=("last_run_unix_ms", "last_result", "last_job_ids"))
    add_object("HealthResponse", "HealthResponse", {
        "status": {"type": "string", "enum": ["ok", "degraded", "maintenance"]},
        "version": ref("NonEmptyString"), "uptime_seconds": ref("NonNegativeInteger"), "model": ref("String"),
        "backend": {"type": "string", "enum": ["mock", "model"]}, "backend_runtime": ref("BackendRuntimeInfo"),
        "serving_profile": ref("ServingProfileDiagnostics"), "http": ref("HttpRuntimeInfo"),
        "model_defaults_profile": ref("ModelDefaultsProfile"), "eval_mode": ref("Boolean"),
        "debug_model_state": ref("Boolean"),
        "default_thinking_enabled": nullable(ref("Boolean")),
        "default_thinking_budget_tokens": nullable(ref("NonNegativeInteger")),
        "default_thinking_budget_ms": nullable(ref("NonNegativeInteger")),
        "fold_reasoning_into_content": ref("Boolean"), "config_hashes": ref("ConfigHashes"),
        "base_weight_identity": nullable(ref("BaseWeightIdentitySummary")),
        "execution_identity": nullable(ref("ExecutionIdentitySummary")),
        "active_adapter": nullable(ref("String")), "loaded_adapter": nullable(ref("String")),
        "loaded_adapter_revision": nullable(ref("String")), "loaded_adapter_count": ref("NonNegativeInteger"),
        "adapters_loaded": ref("NonNegativeInteger"), "request_count": ref("NonNegativeInteger"),
        "requests": ref("RequestMetrics"), "recent_requests": ref("RecentRequestMetrics"),
        "scheduler": nullable(ref("SchedulerStats")), "self_improve_scheduler": ref("SelfImproveSchedulerStatus"),
        "gpu_memory": nullable(ref("GpuMemoryInfo")), "vulkan_buffers": ref("VulkanBufferInfo"),
        "vulkan_buffer_pool": ref("VulkanBufferPoolInfo"),
        "prefix_cache": ref("PrefixCacheHealthInfo"),
        "prompt_caches": ref("PromptCachesInfo"), "decode_runtime": ref("DecodeRuntimeInfo"),
        "prefill_runtime": ref("PrefillRuntimeInfo"), "training": ref("TrainingInfo"),
        "checks": array(ref("HealthCheck"), min_items=6, max_items=6),
    }, "Complete readiness and runtime diagnostic response returned with HTTP 200 or 503.",
        optional=("self_improve_scheduler", "vulkan_buffers", "vulkan_buffer_pool"))

    add_object("BatchingConfigResponse", "BatchingConfigResponse", {
        "configuration": ref("BatchingRuntimeConfig"), "actor_active": ref("Boolean"),
    }, "Resolved batching policy plus live actor state.")
    add_object("PrefixCacheConfig", "config::PrefixCacheConfig", {
        "enabled": ref("Boolean"),
        "max_blocks": nullable(ref("PositiveInteger")),
        "max_entries": nullable(ref("PositiveInteger")),
    }, "Requested startup prefix-cache policy.")
    add_object("PrefixCacheConfigResponse", "PrefixCacheConfigResponse", {
        "configuration": ref("PrefixCacheConfig"),
        "effective_enabled": ref("Boolean"),
        "effective_reason": ref("String"),
        "effective_max_blocks": ref("NonNegativeInteger"),
        "effective_max_entries": ref("NonNegativeInteger"),
        "effective_max_state_bytes": ref("NonNegativeInteger"),
    }, "Requested prefix-cache policy plus backend-qualified live limits.")
    add_object("MemoryTierConfig", "MemoryTierConfig", {
        **{field: ref("NonNegativeInteger") for field in ["total_bytes", "used_bytes", "free_bytes"]},
        **{field: ref("NonNegativeNumber") for field in ["total_gib", "used_gib", "free_gib"]},
    }, "Raw byte and GiB values for one independently constrained memory tier.")
    raw_observation_fields = {
        "probe_failed": ref("Boolean"),
        **{field: nullable(ref("NonNegativeInteger")) for field in [
            "driver_total_bytes", "driver_used_bytes", "driver_free_bytes", "driver_vram_total_bytes",
            "driver_vram_used_bytes", "driver_gtt_total_bytes", "driver_gtt_used_bytes", "host_total_bytes",
            "host_available_bytes", "cgroup_limit_bytes", "cgroup_high_bytes", "cgroup_current_bytes",
            "cgroup_remaining_bytes", "unified_reserve_bytes",
        ]},
        "host_backed": nullable(ref("MemoryTierConfig")),
    }
    add_object("RawMemoryObservations", "RawMemoryObservations", raw_observation_fields,
               "Unmodified driver, host, cgroup, reserve, and host-backed memory observations.")
    add_object("LiveMemoryConfig", "LiveMemoryConfig", {
        **{field: ref("NonNegativeInteger") for field in [
            "total_bytes", "used_bytes", "available_bytes", "effective_capacity_available_bytes",
            "usable_after_governor_floor_bytes", "soft_reserved_bytes", "sample_age_ms", "sample_max_age_ms",
        ]},
        **{field: ref("NonNegativeNumber") for field in [
            "total_gib", "used_gib", "available_gib", "effective_capacity_available_gib",
            "usable_after_governor_floor_gib", "soft_reserved_gib",
        ]},
        "pressure": ref("String"), "source": ref("String"), "sample_stale": ref("Boolean"),
        "sampler_required": ref("Boolean"), "sampler_running": ref("Boolean"), "sampler_healthy": ref("Boolean"),
        "raw_observations": ref("RawMemoryObservations"),
    }, "Live memory view with effective-capacity and governor-floor projections.")
    add_object("MemoryGovernorConfig", "MemoryGovernorConfig", {
        "floor_bytes": ref("NonNegativeInteger"), "floor_gib": ref("NonNegativeNumber"),
        "capacity_limit_bytes": ref("NonNegativeInteger"), "capacity_limit_gib": ref("NonNegativeNumber"),
        "probe_ms": ref("NonNegativeInteger"), "reclaim_mode_requested": ref("String"),
        "reclaim_mode_effective": ref("String"), "reclaim_mode_source": ref("ConfigValueSource"),
        "reclaim_disabled_by_serving_profile": ref("Boolean"),
    }, "Resolved memory-governor floor, capacity, probe, and reclaim state.")
    add_object("VulkanBufferPoolConfig", "VulkanBufferPoolConfig", {
        **{field: ref("NonNegativeInteger") for field in [
            "max_retained_bytes", "retained_bytes", "free_bytes", "borrowed_bytes",
            "eviction_count", "evicted_bytes", "uncached_allocation_count",
            "uncached_allocated_bytes",
        ]},
        "max_retained_gib": ref("NonNegativeNumber"),
        "retained_gib": ref("NonNegativeNumber"),
    }, "Resolved Vulkan scratch-recycler cap, live retention, and pressure-release state.")
    add_object("VramConfig", "VramConfig", {
        "probe_selector": ref("String"), "unified": ref("Boolean"),
        "physical_capacity_bytes": ref("NonNegativeInteger"), "physical_capacity_gib": ref("NonNegativeNumber"),
        "physical_capacity_source": ref("String"), "configured_capacity_bytes": nullable(ref("NonNegativeInteger")),
        "configured_capacity_gib": nullable(ref("NonNegativeNumber")),
        "effective_capacity_bytes": ref("NonNegativeInteger"), "effective_capacity_gib": ref("NonNegativeNumber"),
        "effective_capacity_source": ref("String"), "configured_capacity_clamped": ref("Boolean"),
        "live": ref("LiveMemoryConfig"), "governor": ref("MemoryGovernorConfig"),
        "vulkan_buffer_pool": ref("VulkanBufferPoolConfig"),
    }, "Physical, configured, effective, and live VRAM capacity report.",
        optional=("vulkan_buffer_pool",))
    add_object("KvCacheConfig", "KvCacheConfig", {
        "num_blocks": ref("NonNegativeInteger"), "num_blocks_source": ref("String"),
        "fp8_enabled": ref("Boolean"), "autoscaler": ref("KvAutoscalerState"),
    }, "Resolved KV-cache allocation and autoscaler state.")
    add_object("TrainingOptimizerSupportSchema", "TrainingOptimizerSupportSchema", {
        "id": {"const": "kiln.training-optimizer-support"}, "version": {"const": 1},
    }, "Versioned training optimizer-support report identity.")
    add_object("TrainingOptimizerImplementationConfig", "TrainingOptimizerImplementationConfig", {
        "supported": ref("Boolean"), "route": ref("String"), "native_device_hook": ref("Boolean"),
        "parameter_dtypes": array(ref("String")),
    }, "Backend implementation support for one optimizer kind.")
    add_object("TrainingOptimizerRankConfig", "TrainingOptimizerRankConfig", {
        "minimum": ref("NonNegativeInteger"), "maximum": nullable(ref("NonNegativeInteger")),
        "backend_maximum": nullable(ref("NonNegativeInteger")), "model_maximum": ref("NonNegativeInteger"),
        "live_memory_admission_required": ref("Boolean"),
    }, "LoRA rank limits used by optimizer tuple admission.")
    add_object("TrainingOptimizerTupleConfig", "TrainingOptimizerTupleConfig", {
        "supported": ref("Boolean"), "unavailable_reason": nullable(ref("String")),
        "lora_rank": ref("TrainingOptimizerRankConfig"),
    }, "Resolved support for one optimizer/resident-weight tuple.")
    add_object("TrainingOptimizerKindConfig", "TrainingOptimizerKindConfig", {
        "kind": ref("String"), "backend_implementation": ref("TrainingOptimizerImplementationConfig"),
        "optimizer_tuple": ref("TrainingOptimizerTupleConfig"),
    }, "Implementation and tuple support for one optimizer kind.")
    add_object("TrainingWorkloadSupportConfig", "TrainingWorkloadSupportConfig", {
        "workload": ref("String"), "supported": ref("Boolean"),
        "unavailable_reason": nullable(ref("String")), "allowed_optimizer_kinds": array(ref("String")),
    }, "Resolved optimizer support for one training workload.")
    add_object("TrainingOptimizerSupportConfig", "TrainingOptimizerSupportConfig", {
        "schema": ref("TrainingOptimizerSupportSchema"), "backend": ref("String"), "device": ref("String"),
        "base_weight_dtype": ref("String"), "resolved_lora_parameter_dtype": nullable(ref("String")),
        "immutable_after_startup": ref("Boolean"), "rounding_modes": array(ref("String")),
        "backend_implementation_rounding_modes": array(ref("String")),
        "optimizer_tuple_kinds": array(ref("String")), "workloads": array(ref("TrainingWorkloadSupportConfig")),
        "optimizers": array(ref("TrainingOptimizerKindConfig")),
    }, "Complete backend, dtype, optimizer, workload, and rounding support report.")
    add_object("TrainingConfigResponse", "api::config::TrainingConfig", {
        "runtime_device": nullable(ref("String")), "model_weight_device": ref("String"),
        "native_training_supported": ref("Boolean"), "native_training_unavailable_reason": ref("String"),
        "optimizer_support": nullable(ref("TrainingOptimizerSupportConfig")),
        "checkpoint_policy": ref("GradientCheckpointPolicy"),
        "checkpoint_boundary_policy": ref("CheckpointBoundaryPolicy"),
        "checkpoint_segments": ref("NonNegativeInteger"), "checkpoint_segments_source": ref("String"),
        "checkpointing_enabled": ref("Boolean"),
    }, "Resolved native-training capability and checkpoint configuration.",
        optional=("native_training_unavailable_reason",))
    add_object("MemoryBudgetConfig", "MemoryBudgetConfig", {
        **{field: ref("NonNegativeInteger") for field in [
            "total_vram_bytes", "model_bytes", "kv_cache_bytes", "training_budget_bytes",
        ]},
        **{field: ref("NonNegativeNumber") for field in [
            "total_vram_gib", "model_gib", "kv_cache_gib", "training_budget_gib", "inference_memory_fraction",
        ]},
    }, "Startup model, KV-cache, training, and inference memory budget.")
    add_object("GenerationConfig", "GenerationConfig", {
        "default_thinking_enabled": nullable(ref("Boolean")),
        "default_thinking_budget_tokens": nullable(ref("NonNegativeInteger")),
        "default_thinking_budget_ms": nullable(ref("NonNegativeInteger")),
        "fold_reasoning_into_content": ref("Boolean"),
    }, "Resolved server-wide thinking and reasoning-content defaults.")
    add_object("CheckpointReadPhaseReport", "kiln_model::CheckpointReadPhaseReport", {
        "stage": ref("NonEmptyString"),
        "logical_bytes_completed": ref("NonNegativeInteger"),
        "logical_bytes_total": ref("NonNegativeInteger"),
        "rate_limited_bytes_completed": ref("NonNegativeInteger"),
        "elapsed_milliseconds": ref("NonNegativeInteger"),
        "paced_milliseconds": ref("NonNegativeInteger"),
        "complete": {"const": True},
    }, "Completed logical/read-byte, elapsed-time, and pacing accounting for one checkpoint-read phase.")
    add_object("CheckpointReadReport", "kiln_model::CheckpointReadReport", {
        "configured_bytes_per_second": nullable(ref("PositiveInteger")),
        "snapshot_copy": ref("CheckpointReadPhaseReport"),
        "initial_content_verification": ref("CheckpointReadPhaseReport"),
        "post_upload_content_verification": ref("CheckpointReadPhaseReport"),
        "complete": {"const": True},
    }, "Exact completed accounting for all loader-owned checkpoint-read phases.")
    add_object("CheckpointReadConfigResponse", "CheckpointReadConfigResponse", {
        "configured_mib_per_second": nullable(ref("PositiveInteger")),
        "rate_limited": ref("Boolean"),
        "applicable": ref("Boolean"),
        "not_applicable_reason": nullable({"enum": ["mock_mode"]}),
        "phases": {"const": [
            "snapshot_copy",
            "initial_content_verification",
            "post_upload_content_verification",
        ]},
        "cancellation_poll_milliseconds": {"const": 25},
        "current_work_quantum_interruptible": {"const": False},
        "active_during_inference": {"const": False},
        "restart_required_to_change": {"const": True},
        "observed": nullable(ref("CheckpointReadReport")),
    }, "Resolved startup-only checkpoint-read policy and completed real-model observations.")
    add_object("AcceleratorWeightUploadReport", "kiln_model::AcceleratorWeightUploadReport", {
        "stage": ref("NonEmptyString"),
        "configured_bytes_per_second": nullable(ref("PositiveInteger")),
        "source_bytes_completed": ref("PositiveInteger"),
        "source_bytes_total": ref("PositiveInteger"),
        "source_bytes_reserved": ref("PositiveInteger"),
        "completed_layers": ref("PositiveInteger"),
        "total_layers": ref("PositiveInteger"),
        "reserved_layers": ref("PositiveInteger"),
        "elapsed_milliseconds": ref("NonNegativeInteger"),
        "paced_milliseconds": ref("NonNegativeInteger"),
        "complete": {"const": True},
    }, "Exact completed source-byte, layer, elapsed-time, and pacing accounting for eager base-model accelerator upload.")
    add_object("AcceleratorWeightUploadConfigResponse", "AcceleratorWeightUploadConfigResponse", {
        "configured_mib_per_second": nullable(ref("PositiveInteger")),
        "rate_limited": ref("Boolean"),
        "applicable": ref("Boolean"),
        "not_applicable_reason": nullable({"enum": ["mock_mode", "cpu_device"]}),
        "source_byte_accounting": {"const": "base_model_source_bytes"},
        "cancellation_boundary": {"const": "reserve_before_base_and_each_layer; base_upload_then_transpose_then_pack_then_final"},
        "cancellation_poll_milliseconds": {"const": 25},
        "current_work_quantum_interruptible": {"const": False},
        "active_during_inference": {"const": False},
        "restart_required_to_change": {"const": True},
        "observed": nullable(ref("AcceleratorWeightUploadReport")),
    }, "Resolved startup-only accelerator-weight upload policy and completed real-model observation.")
    add_object("ModelStartupConfigResponse", "ModelStartupConfigResponse", {
        "checkpoint_read": ref("CheckpointReadConfigResponse"),
        "accelerator_weight_upload": ref("AcceleratorWeightUploadConfigResponse"),
    }, "Resolved model-startup resource policy and observations.")
    add_object("SpeculativeBackendMtpConfig", "SpeculativeBackendMtpConfig", {
        "support": ref("String"), "native": ref("Boolean"),
    }, "Selected backend's native MTP capability.")
    add_object("SpeculativeConfig", "SpeculativeConfig", {
        "enabled": ref("Boolean"), "configured_method": ref("SpecMethod"),
        "configured_effective_method": ref("SpecMethod"), "serving_effective_method": ref("SpecMethod"),
        "num_speculative_tokens": ref("NonNegativeInteger"), "draft_layers": ref("NonNegativeInteger"),
        "configured_policy_immutable_after_startup": ref("Boolean"), "serving_routable": ref("Boolean"),
        "serving_unavailable_reason": ref("String"), "draft_token_ceiling": ref("NonNegativeInteger"),
        "backend_mtp": ref("SpeculativeBackendMtpConfig"),
    }, "Configured, effective, and currently routable speculative-decoding state.")
    add_object("OperationalRuntimeConfig", "config::OperationalRuntimeConfig", {
        "bind_host": ref("String"),
        "terminal_access": ref("LocalCapabilityAccess"),
        "terminal_enabled": ref("Boolean"),
        "agent_runs_access": ref("LocalCapabilityAccess"),
        "agent_runs_enabled": ref("Boolean"),
        "pi_bin": nullable(ref("String")),
        "pi_sessions_dir": ref("String"),
        "adapter_library_url": ref("String"),
        "logit_cache_dir": ref("String"),
    }, "Startup-resolved operational paths and local-capability access policy retained by request handlers.")
    add_object("ResolvedApplicationPaths", "config::ResolvedApplicationPaths", {
        "cache_root": ref("String"),
        "cache_root_source": ref("ConfigValueSource"),
        "restart_required_to_change": {"const": True},
    }, "Absolute process-lifetime cache root and the startup authority that selected it.")
    add_object("CudaGraphConfigResponse", "CudaGraphConfigResponse", {
        "requested": ref("Boolean"),
        "capture_allowed_by_serving_profile": ref("Boolean"),
        "effective_policy_enabled": ref("Boolean"),
        "max_cached_graphs": ref("PositiveInteger"),
        "stable_paged_metadata": {"const": True},
        "batched_capture_available": {"const": False},
        "restart_required_to_change": {"const": True},
    }, "Startup-resolved CUDA graph request, cache bound, and fixed safety invariants.")
    add_object("EffectiveConfigurationField", "config::EffectiveConfigurationField", {
        "effective_value": {},
        "source": ref("ConfigValueSource"),
        "canonical_environment": nullable(ref("String")),
        "compatibility_environment": array(ref("String")),
        "redacted": ref("Boolean"),
        "restart_required_to_change": {"const": True},
    }, "One post-precedence typed startup value, its authority, environment spellings, redaction state, and lifecycle.")
    add_object("EffectiveConfiguration", "config::EffectiveConfiguration", {
        "schema_id": {"const": "kiln.effective-configuration.v1"},
        "schema_version": {"const": 1},
        "effective_config_hash": nullable(ref("Sha256")),
        "fixed_field_count": {"const": 118},
        "dynamic_field_count": ref("NonNegativeInteger"),
        "all_fields_restart_required_to_change": {"const": True},
        "fields": {
            "type": "object",
            "propertyNames": {
                "pattern": "^[A-Za-z0-9_-]+(?:\\.[A-Za-z0-9_-]+)+$",
            },
            "additionalProperties": ref("EffectiveConfigurationField"),
            "x-kiln-fixed-field-count": 118,
            "x-kiln-dynamic-field-templates": [
                "teachers.credentials.<id>.origin",
                "teachers.credentials.<id>.api_key_env",
            ],
        },
    }, "The deterministic flat map of every typed startup leaf after precedence, plus dynamic teacher-credential leaves.")
    add_object("ConfigResponse", "ConfigResponse", {
        "effective_configuration": ref("EffectiveConfiguration"),
        "serving_profile": ref("ServingProfileDiagnostics"),
        "accelerator_runtime": ref("ResolvedAcceleratorRuntimePolicy"),
        "rocm_graphs": nullable(ref("RocmGraphStats")),
        "rocm_graphs_unavailable_reason": nullable(ref("RocmGraphUnavailableReason")),
        "rocm_graph_telemetry": nullable(ref("RocmGraphLiveTelemetry")),
        "rocm_graph_telemetry_unavailable_reason": nullable(ref("RocmGraphUnavailableReason")),
        "cuda_graphs": ref("CudaGraphConfigResponse"),
        "decode_runtime": ref("DecodeRuntimeConfig"), "batching": ref("BatchingConfigResponse"),
        "prefix_cache": ref("PrefixCacheConfigResponse"),
        "streaming_prefill": ref("StreamingPrefillRuntimeConfig"), "speculative": ref("SpeculativeConfig"),
        "model_startup": ref("ModelStartupConfigResponse"),
        "paths": ref("ResolvedApplicationPaths"),
        "vram": ref("VramConfig"), "kv_cache": ref("KvCacheConfig"),
        "training": ref("TrainingConfigResponse"), "memory_budget": ref("MemoryBudgetConfig"),
        "generation": ref("GenerationConfig"), "operational": ref("OperationalRuntimeConfig"),
    }, "Complete resolved configuration and live capacity response.")

    add_object("LoadedAdapterDebugState", "LoadedAdapterDebugState", {
        "name": ref("String"), "path": ref("String"), "adapter_model_sha256": nullable(ref("Sha256")),
    }, "One loaded adapter's name, path, and optional content identity.")
    add_object("AdapterDebugState", "AdapterDebugState", {
        "adapter_dir": ref("String"), "active_adapter": nullable(ref("String")),
        "loaded_adapter": nullable(ref("String")), "loaded_adapter_revision": nullable(ref("String")),
        "loaded_adapters": array(ref("LoadedAdapterDebugState")),
        "available_adapter_count": ref("NonNegativeInteger"), "load_errors": mapping(ref("String")),
    }, "Active, loaded, discoverable, and failed adapter state.")
    add_object("ModelDebugState", "ModelDebugState", {
        "path": nullable(ref("String")), "served_model_id": ref("String"),
        "defaults_profile": ref("ModelDefaultsProfile"), "num_layers": ref("NonNegativeInteger"),
        "num_attention_heads": ref("NonNegativeInteger"), "num_kv_heads": ref("NonNegativeInteger"),
        "max_position_embeddings": ref("NonNegativeInteger"),
        "base_weight_shard_manifest": nullable(ref("BaseWeightShardManifest")),
        "execution_provenance": nullable(ref("ExecutionProvenanceV1")),
        "backend_runtime": ref("BackendRuntimeInfo"),
    }, "Loaded model shape, provenance, defaults, and backend health state.")
    add_object("BatchingEngineDebugState", "BatchingEngineDebugState", {
        "backend": ref("String"), "enabled": ref("Boolean"), "configuration": ref("BatchingRuntimeConfig"),
        "snapshot": nullable(ref("BatchingEngineSnapshot")),
    }, "Selected batching backend, resolved configuration, and live execution state.")
    add_object("ThinkingDebugState", "ThinkingDebugState", {
        "eval_mode": ref("Boolean"), "default_thinking_enabled": nullable(ref("Boolean")),
        "default_thinking_budget_tokens": nullable(ref("NonNegativeInteger")),
        "default_thinking_budget_ms": nullable(ref("NonNegativeInteger")),
        "profile_server_default_thinking_enabled": nullable(ref("Boolean")),
        "template_default_thinking_enabled": ref("Boolean"),
        "eval_mode_default_thinking_enabled": ref("Boolean"),
    }, "Resolved thinking defaults from request-independent startup policy.")
    add_object("BatchedStateCacheStats", "BatchedStateCacheStats", {
        "entry_present": ref("Boolean"),
        "capacity_rows": ref("NonNegativeInteger"),
        "logical_rows": ref("NonNegativeInteger"),
        "resident": ref("Boolean"),
        "active_leases": ref("NonNegativeInteger"),
        "max_active_leases": ref("NonNegativeInteger"),
        "take_hit_count": ref("NonNegativeInteger"),
        "take_miss_count": ref("NonNegativeInteger"),
        "take_miss_while_leased_count": ref("NonNegativeInteger"),
        "exact_reuse_count": ref("NonNegativeInteger"),
        "resident_capacity_reuse_count": ref("NonNegativeInteger"),
        "resident_prefix_view_count": ref("NonNegativeInteger"),
        "resident_refresh_count": ref("NonNegativeInteger"),
        "fresh_assembly_count": ref("NonNegativeInteger"),
        "rejected_missing_row_ids_count": ref("NonNegativeInteger"),
        "rejected_nonresident_rows_count": ref("NonNegativeInteger"),
        "rejected_nonresident_cache_count": ref("NonNegativeInteger"),
        "rejected_insufficient_capacity_count": ref("NonNegativeInteger"),
        "park_count": ref("NonNegativeInteger"),
        "park_replacement_eviction_count": ref("NonNegativeInteger"),
        "explicit_invalidation_count": ref("NonNegativeInteger"),
        "explicit_invalidation_eviction_count": ref("NonNegativeInteger"),
        "completed_row_preservation_count": ref("NonNegativeInteger"),
        "completed_row_eviction_count": ref("NonNegativeInteger"),
        "lease_drop_eviction_count": ref("NonNegativeInteger"),
        "resident_prefix_snapshot_suppression_count": ref("NonNegativeInteger"),
    }, "Current ownership and process-lifetime lifecycle counters for the parked batched recurrent-state cache.")
    add_object("GdnRecurrentStateResidencyStats", "GdnRecurrentStateResidencyStats", {
        "entry_count": ref("NonNegativeInteger"),
        "buffer_bytes": ref("NonNegativeInteger"),
        "allocation_bytes": ref("NonNegativeInteger"),
    }, "Current direct backend-private GDN recurrent-state ownership across resumable prefill and scoped decode rows; the persistent batched cache is separate.")
    add_object("CacheDebugState", "CacheDebugState", {
        "deterministic_completion_entries": ref("NonNegativeInteger"),
        "deterministic_chat_request_entries": ref("NonNegativeInteger"),
        "deterministic_chat_choices_entries": ref("NonNegativeInteger"),
        "deterministic_batch_entries": ref("NonNegativeInteger"),
        "batched_recurrent_state": ref("BatchedStateCacheStats"),
        "resident_recurrent_state": ref("GdnRecurrentStateResidencyStats"),
        "rendered_prompt": ref("PromptCacheInfo"), "prompt_token": ref("PromptCacheInfo"),
        "prefix_cache": ref("PrefixCacheInfo"),
    }, "Deterministic response, prompt, token, prefix-cache, and backend recurrent-state ownership.")
    add_object("TrainingDebugState", "TrainingDebugState", {
        "checkpoint_boundary_policy": ref("CheckpointBoundaryPolicy"),
    }, "Training checkpoint-boundary state published by the debug endpoint.")
    add_object("ModelStateResponse", "ModelStateResponse", {
        "model": ref("ModelDebugState"), "adapters": ref("AdapterDebugState"),
        "config_hashes": ref("ConfigHashes"), "http": ref("HttpRuntimeInfo"),
        "decode_runtime": ref("DecodeRuntimeConfig"),
        "accelerator_runtime": ref("ResolvedAcceleratorRuntimePolicy"),
        "cuda_graphs": ref("CudaGraphConfigResponse"),
        "cuda_synchronization": ref("CudaSynchronizationRuntimeStats"),
        "rocm_synchronization": ref("RocmSynchronizationRuntimeStats"),
        "rocm_graphs": nullable(ref("RocmGraphStats")),
        "rocm_graphs_unavailable_reason": nullable(ref("RocmGraphUnavailableReason")),
        "rocm_graph_telemetry": nullable(ref("RocmGraphLiveTelemetry")),
        "rocm_graph_telemetry_unavailable_reason": nullable(ref("RocmGraphUnavailableReason")),
        "kv_autoscaler": ref("KvAutoscalerState"), "streaming_prefill": ref("StreamingPrefillRuntimeConfig"),
        "training": ref("TrainingDebugState"),
        "batching_engine": ref("BatchingEngineDebugState"), "thinking": ref("ThinkingDebugState"),
        "caches": ref("CacheDebugState"),
    }, "Complete opt-in model, adapter, provenance, typed runtime, and cache debug state.")
    add_object("DebugDisabledResponse", "DebugDisabledResponse", {
        "error": {"const": "debug endpoint disabled"},
        "enable_with": {"const": "set server.debug_model_state=true or server.eval_mode=true"},
    }, "HTTP 403 body returned when model-state diagnostics are not enabled.")
    add_object("DebugProvenanceErrorResponse", "serde_json::Value", {
        "error": {"const": "invalid execution provenance"}, "detail": ref("NonEmptyString"),
    }, "HTTP 500 body returned when the resident execution provenance fails validation.")

    add_object("ModelInfo", "ModelInfo", {
        "id": ref("String"), "object": {"const": "model"}, "owned_by": {"const": "kiln"},
    }, "One OpenAI-compatible served model descriptor.")
    add_object("ModelsResponse", "ModelsResponse", {
        "object": {"const": "list"},
        "data": array(ref("ModelInfo"), min_items=1, max_items=1),
    }, "OpenAI-compatible list of served models.")
    add_object(
        "LatencyStallReasonCounts",
        "LatencyStallReasonCounts",
        {field: ref("NonNegativeInteger") for field in LATENCY_STALL_REASON_FIELDS},
        "Fixed-cardinality token-stall counts by dominant blocking reason.",
    )
    nullable_duration = nullable(ref("NonNegativeNumber"))
    add_object(
        "LatencyPhaseTimings",
        "LatencyPhaseTimings",
        {field: nullable_duration for field in LATENCY_PHASE_FIELDS},
        "Bounded request phase timings; null means the serving path did not measure that subphase.",
    )
    nullable_number = nullable(ref("NonNegativeNumber"))
    add_object("RequestLatencyDiagnostics", "RequestLatencyDiagnostics", {
        "emitted_tokens": ref("NonNegativeInteger"), "gap_samples": ref("NonNegativeInteger"),
        "retained_gap_samples": {"type": "integer", "minimum": 0, "maximum": 8192},
        "gap_samples_truncated": ref("Boolean"),
        "ttft_ms": nullable_number, "itl_ms_p50": nullable_number,
        "itl_ms_p99": nullable_number, "itl_ms_p999": nullable_number,
        "max_itl_ms": nullable_number, "stall_threshold_ms": nullable_number,
        "stall_count": ref("NonNegativeInteger"), "unexplained_stall_count": ref("NonNegativeInteger"),
        "stall_reasons": ref("LatencyStallReasonCounts"), "phases": ref("LatencyPhaseTimings"),
    }, "Request-local TTFT, ITL tail, bounded stall reasons, and honest phase-coverage diagnostics.")
    add_object("DecodeStatsSnapshot", "DecodeStatsSnapshot", {
        "tok_per_sec": ref("NonNegativeNumber"), "p50_itl_ms": ref("NonNegativeNumber"),
        "p99_itl_ms": ref("NonNegativeNumber"), "p999_itl_ms": ref("NonNegativeNumber"),
        "mean_itl_ms": ref("NonNegativeNumber"), "max_itl_ms": ref("NonNegativeNumber"),
        "stall_threshold_ms": ref("NonNegativeNumber"), "stall_count": ref("NonNegativeInteger"),
        "unexplained_stall_count": ref("NonNegativeInteger"),
        "stall_reasons": ref("LatencyStallReasonCounts"),
        "sample_count": ref("NonNegativeInteger"), "window_secs": {"const": 60.0},
    }, "Rolling request-local decode throughput, ITL tail, and bounded stall diagnosis.")
    add_object("RequestThinkingBudget", "RequestThinkingBudget", {
        "configured": ref("Boolean"), "max_tokens": ref("NonNegativeInteger"),
        "max_time_ms": ref("NonNegativeInteger"), "tokens_source": ref("ThinkingBudgetSource"),
        "time_source": ref("ThinkingBudgetSource"), "applied": ref("Boolean"),
        "triggered": ref("Boolean"), "trigger": ref("String"), "closed": ref("Boolean"),
        "thinking_tokens": ref("NonNegativeInteger"), "thinking_time_ms": ref("NonNegativeInteger"),
    }, "Configured thinking budget and observed request outcome.", optional=(
        "max_tokens", "max_time_ms", "applied", "triggered", "trigger", "closed",
        "thinking_tokens", "thinking_time_ms",
    ))
    add_object("RequestRecord", "RequestRecord", {
        "id": ref("String"), "timestamp_unix_ms": ref("NonNegativeInteger"), "model": ref("String"),
        "prompt_preview": ref("String"), "completion_preview": ref("String"),
        "prompt_tokens": ref("NonNegativeInteger"), "completion_tokens": ref("NonNegativeInteger"),
        "duration_ms": ref("NonNegativeInteger"), "streamed": ref("Boolean"), "finish_reason": ref("String"),
        "adapter": ref("String"), "temperature": ref("FiniteNumber"), "top_p": ref("FiniteNumber"),
        "max_tokens": ref("NonNegativeInteger"), "ttft_ms": ref("NonNegativeInteger"),
        "model_prefill_ms": ref("NonNegativeInteger"), "model_decode_ms": ref("NonNegativeInteger"),
        "error": ref("String"), "thinking_mode": ref("String"), "prefix_cache": ref("String"),
        "prompt_full": ref("String"), "completion_full": ref("String"), "user_agent": ref("String"),
        "client": ref("String"), "thinking_budget": ref("RequestThinkingBudget"),
        "latency": ref("RequestLatencyDiagnostics"),
    }, (
        "One newest-first bounded recent-request record. The record can contain full prompt and "
        "completion text and therefore belongs inside the server's trusted access boundary."
    ), optional=(
        "adapter", "temperature", "top_p", "max_tokens", "ttft_ms", "model_prefill_ms",
        "model_decode_ms", "error", "thinking_mode", "prefix_cache", "prompt_full",
        "completion_full", "user_agent", "client", "thinking_budget", "latency",
    ))
    add_definition("Vec_RequestRecord", "Vec<RequestRecord>", {
        "type": "array", "items": ref("RequestRecord"),
    }, "Newest-first snapshot of the bounded recent-request ring.")
    add_object("CacheStats", "CacheStats", {
        "total_entries": ref("NonNegativeInteger"), "total_bytes": ref("NonNegativeInteger"),
        "per_teacher": mapping(ref("NonNegativeInteger")),
    }, "Persistent logit-cache occupancy keyed by authoritative teacher revision.")
    add_object("CacheStatsResponse", "CacheStatsResponse", {
        "root": ref("String"), "stats": ref("CacheStats"),
    }, "Persistent logit-cache root and occupancy report.")


def example_for(schema: dict[str, Any]) -> Any:
    if not schema:
        return None
    if "$ref" in schema:
        return example_for(DEFS[schema["$ref"].split("/")[-1]])
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    if "oneOf" in schema:
        return example_for(schema["oneOf"][0])
    if "anyOf" in schema:
        non_null = next((item for item in schema["anyOf"] if item.get("type") != "null"), schema["anyOf"][0])
        return example_for(non_null)
    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties", {})
        return {name: example_for(properties[name]) for name in schema.get("required", [])}
    if schema_type == "array":
        return [example_for(schema["items"])] * schema.get("minItems", 0)
    if schema_type == "string":
        if schema.get("pattern", "").startswith("^sha256:"):
            return "sha256:" + "0" * 64
        return "value"
    if schema_type in ("integer", "number"):
        return schema.get("minimum", 0)
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None
    raise ValueError(f"cannot generate example for {schema}")


ENTRYPOINTS = (
    "CacheStatsResponse",
    "ConfigResponse",
    "DebugDisabledResponse",
    "DebugProvenanceErrorResponse",
    "DecodeStatsSnapshot",
    "HealthResponse",
    "ModelStateResponse",
    "ModelsResponse",
    "RequestRecord",
    "Vec_RequestRecord",
)


def build_schema() -> dict[str, Any]:
    build_definitions()
    apply_editorial_descriptions()
    examples = {name: [example_for(ref(name))] for name in ENTRYPOINTS}
    examples["ModelsResponse"] = [{
        "object": "list",
        "data": [{"id": "Qwen3.5-4B", "object": "model", "owned_by": "kiln"}],
    }]
    examples["DecodeStatsSnapshot"] = [{
        "tok_per_sec": 42.5, "p50_itl_ms": 23.1, "p99_itl_ms": 31.8,
        "p999_itl_ms": 48.2, "mean_itl_ms": 24.0, "max_itl_ms": 51.0,
        "stall_threshold_ms": 250.0, "stall_count": 0, "unexplained_stall_count": 0,
        "stall_reasons": {
            "actor_queue": 0, "actor_admission": 0, "actor_prefill": 0, "actor_decode": 0,
            "actor_cycle_idle": 0,
            "response_delivery": 0, "handler_queue": 0, "client_delivery": 0, "unexplained": 0,
            "sampling": 0, "readback": 0, "gpu_lock_wait": 0, "graph_capture": 0,
            "graph_replay": 0, "synchronization": 0, "resize": 0, "trim": 0,
            "adapter": 0, "training": 0,
        },
        "sample_count": 84, "window_secs": 60.0,
    }]
    examples["RequestRecord"] = [{
        "id": "chatcmpl-example", "timestamp_unix_ms": 1_800_000_000_000,
        "model": "Qwen3.5-4B", "prompt_preview": "Explain bounded thinking.",
        "completion_preview": "A bounded thinking request...", "prompt_tokens": 12,
        "completion_tokens": 48, "duration_ms": 730, "streamed": True, "finish_reason": "stop",
        "ttft_ms": 88, "model_prefill_ms": 41, "model_decode_ms": 620,
        "thinking_mode": "enabled", "prefix_cache": "miss", "client": "dashboard",
        "thinking_budget": {
            "configured": True, "max_tokens": 32, "tokens_source": "request",
            "time_source": "unlimited", "applied": True, "triggered": True,
            "trigger": "tokens", "closed": True, "thinking_tokens": 32, "thinking_time_ms": 410,
        },
        "latency": {
            "emitted_tokens": 48, "gap_samples": 47, "retained_gap_samples": 47,
            "gap_samples_truncated": False, "ttft_ms": 88.0, "itl_ms_p50": 12.0,
            "itl_ms_p99": 31.8, "itl_ms_p999": 35.0, "max_itl_ms": 36.0,
            "stall_threshold_ms": 250.0, "stall_count": 0, "unexplained_stall_count": 0,
            "stall_reasons": {
                "actor_queue": 0, "actor_admission": 0, "actor_prefill": 0, "actor_decode": 0,
                "actor_cycle_idle": 0,
                "response_delivery": 0, "handler_queue": 0, "client_delivery": 0, "unexplained": 0,
                "sampling": 0, "readback": 0, "gpu_lock_wait": 0, "graph_capture": 0,
                "graph_replay": 0, "synchronization": 0, "resize": 0, "trim": 0,
                "adapter": 0, "training": 0,
            },
            "phases": {
                "actor_queue_ms": 8.0, "actor_admission_ms": 1.0, "tokenization_ms": 2.0,
                "prefill_ms": 39.0, "decode_ms": 580.0, "actor_cycle_idle_ms": 0.0,
                "sampling_ms": None,
                "readback_ms": None, "response_delivery_ms": 3.0, "handler_queue_ms": 1.0,
                "client_delivery_ms": 2.0, "gpu_lock_wait_ms": 0.7, "graph_capture_ms": None,
                "graph_replay_ms": None, "synchronization_ms": 1.4, "resize_ms": None,
                "trim_ms": None, "adapter_ms": None, "training_ms": None, "unexplained_ms": 1.0,
            },
        },
    }]
    examples["Vec_RequestRecord"] = [examples["RequestRecord"]]
    examples["CacheStatsResponse"] = [{
        "root": "/var/lib/kiln/logit-cache",
        "stats": {"total_entries": 12, "total_bytes": 4096, "per_teacher": {"teacher-revision": 12}},
    }]
    health = examples["HealthResponse"][0]
    health.update({"status": "ok", "version": "0.1.0", "model": "Qwen3.5-4B (32L, 32H, 8KV)"})
    health["vulkan_buffers"] = {
        "live_device_local_buffers": 96,
        "live_device_local_bytes": 8_589_934_592,
        "live_host_visible_buffers": 24,
        "live_host_visible_bytes": 67_108_864,
        "peak_live_bytes": 8_724_152_320,
        "device_local_allocations": 140,
        "device_local_allocated_bytes": 9_663_676_416,
        "device_local_frees": 44,
        "device_local_freed_bytes": 1_073_741_824,
        "host_visible_allocations": 88,
        "host_visible_allocated_bytes": 201_326_592,
        "host_visible_frees": 64,
        "host_visible_freed_bytes": 134_217_728,
    }
    health["vulkan_buffer_pool"] = {
        "max_retained_bytes": 3_221_225_472,
        "bucket_count": 48,
        "buffer_count": 180,
        "retained_bytes": 2_148_204_544,
        "free_buffer_count": 176,
        "free_bytes": 2_080_768_000,
        "borrowed_buffer_count": 4,
        "borrowed_bytes": 67_436_544,
        "cache_hits": 981_000,
        "cache_misses": 180,
        "eviction_count": 0,
        "evicted_bytes": 0,
        "uncached_allocation_count": 0,
        "uncached_allocated_bytes": 0,
    }
    health["backend_runtime"].update({"healthy": True, "quarantined": False, "restart_required": False})
    health["serving_profile"].update({"profile": "stable", "immutable_after_startup": True})
    health["serving_profile"]["effective_policy"].update({
        "inference_admission": True,
        "training_gpu_ownership": True,
        "adapter_weight_transitions": True,
        "dynamic_kv_resize": True,
        "allocator_reclaim": True,
        "live_graph_capture": True,
        "exclusive_gpu_behavior": "writer_priority",
    })
    health["model_defaults_profile"].update({
        "name": "Qwen3.5-4B",
        "canonical_model_id": "Qwen/Qwen3.5-4B",
        "canonical_served_model_id": "Qwen3.5-4B",
    })
    health["checks"] = [
        {"name": name, "pass": True}
        for name in [
            "model_loaded",
            "scheduler_responsive",
            "backend_runtime_healthy",
            "inference_admission",
            "inference_prewarm_complete",
            "execution_provenance_valid",
        ]
    ]
    examples["DebugDisabledResponse"] = [{
        "error": "debug endpoint disabled",
        "enable_with": "set server.debug_model_state=true or server.eval_mode=true",
    }]
    examples["DebugProvenanceErrorResponse"] = [{
        "error": "invalid execution provenance",
        "detail": "execution provenance digest mismatch",
    }]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://ericflo.github.io/kiln/contracts/kiln-observability-v1.schema.json",
        "title": "Kiln Read-only Serving and Observability API v1",
        "description": (
            "Canonical read-only response schemas for health and readiness, resolved configuration, "
            "opt-in model-state diagnostics, model discovery, decode statistics, recent requests, and "
            "persistent teacher-logit cache statistics. Kiln does not authenticate these endpoints. "
            "`GET /v1/stats/recent-requests` can return stored prompt and completion text, so keep the "
            "server on loopback or behind an authenticated reverse proxy."
        ),
        **STATUS,
        "x-kiln-entrypoints": list(ENTRYPOINTS),
        "x-kiln-examples": examples,
        "x-kiln-health-http-status": {
            "200": {"status": "ok"},
            "503": {"status": ["degraded", "maintenance"]},
        },
        "oneOf": [ref(name) for name in ENTRYPOINTS],
        "$defs": {name: DEFS[name] for name in sorted(DEFS)},
    }


def render() -> str:
    DEFS.clear()
    return json.dumps(build_schema(), indent=2, ensure_ascii=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when the checked-in schema is stale")
    args = parser.parse_args()
    expected = render()
    if args.check:
        observed = OUTPUT.read_text() if OUTPUT.exists() else ""
        if observed != expected:
            print(f"{OUTPUT.relative_to(ROOT)} is stale; run {Path(__file__).relative_to(ROOT)}", flush=True)
            return 1
        print(f"observability schema generation passed: {len(DEFS)} closed definitions")
        return 0
    OUTPUT.write_text(expected)
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(DEFS)} closed definitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
