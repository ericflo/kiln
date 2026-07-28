#!/usr/bin/env python3
"""Run a backend-profiled continuous mixed-load soak with bounded evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal

import serve_mixed_load as mixed


ROOT = Path(__file__).resolve().parents[2]
RESULT_ENV = "KILN_QUALIFICATION_CASE_RESULT"
VARIANT_ENV = "KILN_QUALIFICATION_VARIANT_ID"
CASE_ID = "continuous-mixed-load"
RUNTIME_VARIANT = "autoscale-off"
ROCM_ENDURANCE_VARIANT = "rocm-endurance"
VULKAN_RUNTIME_VARIANT = "vulkan-development-soak"
VULKAN_ENDURANCE_VARIANT = "vulkan-endurance"
CUDA_ENDURANCE_VARIANT = "cuda-endurance"
VULKAN_MAX_PREFILL_TOKENS_PER_CYCLE = 128
VULKAN_QUALIFIED_ACTIVE_REQUESTS = 4
VULKAN_QUALIFIED_DECODE_BATCH = 2
VULKAN_QUALIFIED_PREFILL_ADMISSION_QUANTUM = 2
VULKAN_MAX_PREFILL_LAYERS_PER_CYCLE = 4
VULKAN_QUALIFIED_BUFFER_POOL_GB = 3.5
VULKAN_QUALIFIED_WAVE_CONCURRENCY = (1, 4)
VULKAN_QUALIFIED_PROMPT_WORDS = (16, 32, 64, 96)
SLOT_BY_REQUEST = "slot_by_request"
COHORT_BY_CYCLE = "cohort_by_cycle"
PromptAssignment = Literal["slot_by_request", "cohort_by_cycle"]
DEFAULT_MEMORY_GROWTH_LIMIT_BYTES = 512 * 1024 * 1024
WAVE_CONCURRENCY = (1, 8, 12, 8)
PROMPT_WORDS = (16, 32, 64, 128, 256, 384, 512, 768, 96, 192, 1024, 48)
MAX_TOKENS = 32
CANCEL_EVERY_WAVES = 5
CANCELLATION_MAX_TOKENS = 512
CANCELLATION_PROMPT_WORDS = 48
ROCM_REQUEST_TIMEOUT_SECONDS = 120.0
QUALIFICATION_DURATION_SECONDS = 1800.0
ROCM_ENDURANCE_DURATION_SECONDS = 24 * 60 * 60.0
VULKAN_ENDURANCE_DURATION_SECONDS = 8 * 60 * 60.0
CUDA_ENDURANCE_DURATION_SECONDS = 8 * 60 * 60.0
CUDA_QUALIFIED_WAVE_CONCURRENCY = (1, 4)
CUDA_QUALIFIED_PROMPT_WORDS = (16, 32, 64, 96)
CUDA_STABILIZATION_MIN_ROTATIONS = 2
CUDA_STABILIZATION_MAX_ROTATIONS = 4
CASE_TEARDOWN_GRACE_SECONDS = 180.0
REQUEST_WORKER_CLEANUP_TIMEOUT_SECONDS = 10.0
MAX_STEADY_STATE_WARMUP_WAVES = 16
ROCM_GRAPH_ADMISSION_POLICY = "idle_owner_lru_then_active_fair_lru"
ROCM_GRAPH_ACTIVE_OWNER_FLOOR = 1
# Fair active relief creates transition room only when a candidate needs it.
ROCM_GRAPH_TRANSITION_HEADROOM_ENTRIES = 0
ROCM_GRAPH_CACHE_MAX = (
    mixed.MAX_ACTIVE_REQUESTS * ROCM_GRAPH_ACTIVE_OWNER_FLOOR
    + ROCM_GRAPH_TRANSITION_HEADROOM_ENTRIES
)
MIN_STABILIZATION_CYCLES = 4
MAX_STABILIZATION_CYCLES = 8
REQUIRED_STABLE_CYCLES = 2
STABILIZATION_GPU_DELTA_LIMIT_BYTES = 64 * 1024 * 1024
STABILIZATION_RSS_DELTA_LIMIT_BYTES = 16 * 1024 * 1024
VULKAN_ACTIVE_GPU_PEAK_GROWTH_LIMIT_BYTES = 1024 * 1024 * 1024
SETUP_DEADLINE_SECONDS = 1200.0
HOST_GUARD_POLL_INTERVAL_SECONDS = 0.25
HOST_MEMORY_AVAILABLE_FLOOR_BYTES = 8 * 1024 * 1024 * 1024
HOST_SWAP_GROWTH_LIMIT_BYTES = 512 * 1024 * 1024
ACCELERATOR_TELEMETRY_ACTIVE_BUSY_FLOOR_PERCENT = 50
AMD_GPU_VENDOR_ID = "0x1002"
PROCESS_MEMORY_MAPPING_CATEGORIES = (
    "anonymous",
    "device",
    "file",
    "heap",
    "kernel",
    "shared_memory",
    "stack",
)
PROCESS_MEMORY_MAPPING_FIELDS = (
    "Size",
    "Rss",
    "Pss",
    "Anonymous",
    "AnonHugePages",
    "Private_Dirty",
    "Swap",
)
SMAPS_HEADER = re.compile(
    r"^(?P<range>[0-9A-Fa-f]+-[0-9A-Fa-f]+)\s+"
    r"[r-][w-][x-][ps]\s+[0-9A-Fa-f]+\s+"
    r"[0-9A-Fa-f]+:[0-9A-Fa-f]+\s+\d+(?:\s+(?P<pathname>.*))?$"
)

def _vulkan_variant_config() -> dict[str, Any]:
    config = mixed._variant_config(
        serving_profile="experimental",
        kv_autoscale_requested=False,
        kv_autoscale_enabled=False,
        memory_reclaim_requested_mode="off",
        memory_reclaim_mode="off",
        rocm_graphs_requested=False,
        rocm_graphs_enabled=False,
        request_timeout_seconds=600,
    )
    config["build"] = mixed.VULKAN_BUILD_SPEC.effective_config()
    config["server"]["max_prefill_tokens_per_cycle"] = (
        VULKAN_MAX_PREFILL_TOKENS_PER_CYCLE
    )
    config["server"]["max_prefill_layers_per_cycle"] = (
        VULKAN_MAX_PREFILL_LAYERS_PER_CYCLE
    )
    config["server"].update(
        {
            "max_decode_batch": VULKAN_QUALIFIED_DECODE_BATCH,
            "max_active_requests": VULKAN_QUALIFIED_ACTIVE_REQUESTS,
            "max_prefill_staging_slots": (
                VULKAN_QUALIFIED_PREFILL_ADMISSION_QUANTUM
            ),
        }
    )
    config["batching"] = {
        "actor_cycle_idle_ms": 0,
        "prefill_admission_quantum": VULKAN_QUALIFIED_PREFILL_ADMISSION_QUANTUM,
    }
    config["model"] = {
        "checkpoint_read_mib_per_second": mixed.CHECKPOINT_READ_MIB_PER_SECOND,
        "accelerator_weight_upload_mib_per_second": (
            mixed.ACCELERATOR_WEIGHT_UPLOAD_MIB_PER_SECOND
        ),
        "vulkan_decode_weight_prewarm": mixed.VULKAN_DECODE_WEIGHT_PREWARM,
        "vulkan_decode_weight_prewarm_mib_per_second": (
            mixed.VULKAN_DECODE_WEIGHT_PREWARM_MIB_PER_SECOND
        ),
    }
    config["runtime"].update(
        {
            "prefix_cache_requested_enabled": True,
            "prefix_cache_effective_enabled": False,
            "prefix_cache_effective_reason": "vulkan_correctness_quarantine",
        }
    )
    config["runtime"]["vulkan_buffer_pool_gb"] = VULKAN_QUALIFIED_BUFFER_POOL_GB
    return config


CUDA_BUILD_SPEC = mixed.SourceBuildSpec(
    backend="CUDA",
    features="cuda",
    qualification_device_required=True,
)


def _cuda_variant_config() -> dict[str, Any]:
    config = mixed._variant_config(
        serving_profile="stable",
        kv_autoscale_requested=False,
        kv_autoscale_enabled=False,
        memory_reclaim_requested_mode="off",
        memory_reclaim_mode="off",
        rocm_graphs_requested=False,
        rocm_graphs_enabled=False,
        request_timeout_seconds=600,
        max_decode_batch=4,
    )
    config["build"] = CUDA_BUILD_SPEC.effective_config()
    config["server"].update(
        {
            "max_prefill_staging_slots": 0,
            "max_active_requests": config["server"]["max_decode_batch"],
            "max_prefill_staging_priority_burst": 0,
        }
    )
    config["runtime"].update(
        {
            "prefix_cache_requested_enabled": False,
            "prefix_cache_effective_enabled": False,
            "prefix_cache_effective_reason": "cuda_prefill_semantics_quarantine",
            "vulkan_buffer_pool_gb": 0.0,
        }
    )
    return config


mixed.VARIANT_CONFIGS[ROCM_ENDURANCE_VARIANT] = mixed.VARIANT_CONFIGS[RUNTIME_VARIANT]
mixed.VARIANT_CONFIGS[VULKAN_RUNTIME_VARIANT] = _vulkan_variant_config()
mixed.VARIANT_CONFIGS[VULKAN_ENDURANCE_VARIANT] = _vulkan_variant_config()
mixed.VARIANT_CONFIGS[CUDA_ENDURANCE_VARIANT] = _cuda_variant_config()


@dataclasses.dataclass(frozen=True)
class SoakRuntime:
    variant_id: str
    backend: str
    build_spec: mixed.SourceBuildSpec
    gpu_memory_scope: str
    gpu_memory_source: str
    graph_execution_required: bool
    wave_concurrency: tuple[int, ...]
    prompt_words: tuple[int, ...]
    prompt_assignment: PromptAssignment
    max_tokens: int
    cancel_every_waves: int
    cancellation_max_tokens: int
    cancellation_prompt_words: int
    request_timeout_seconds: float
    max_steady_state_warmup_waves: int
    graph_cache_max: int
    min_stabilization_cycles: int
    max_stabilization_cycles: int
    required_stable_cycles: int
    stabilization_gpu_delta_limit_bytes: int
    stabilization_rss_delta_limit_bytes: int
    active_gpu_peak_growth_limit_bytes: int | None
    vulkan_allocation_growth_limit_count: int | None
    vulkan_buffer_pool_gb: float
    setup_deadline_seconds: float
    host_mem_available_floor_bytes: int | None = None
    host_swap_growth_limit_bytes: int | None = None
    inference_memory_fraction: float | None = None
    memory_floor_gb: float | None = None
    accelerator_telemetry_enabled: bool = True
    accelerator_telemetry_required: bool = False


ROCM_RUNTIME = SoakRuntime(
    variant_id=RUNTIME_VARIANT,
    backend="rocm",
    build_spec=mixed.ROCM_BUILD_SPEC,
    gpu_memory_scope="device_global",
    gpu_memory_source='server_metrics:kiln_gpu_memory_bytes{kind="used"}',
    graph_execution_required=True,
    wave_concurrency=WAVE_CONCURRENCY,
    prompt_words=PROMPT_WORDS,
    prompt_assignment=SLOT_BY_REQUEST,
    max_tokens=MAX_TOKENS,
    cancel_every_waves=CANCEL_EVERY_WAVES,
    cancellation_max_tokens=CANCELLATION_MAX_TOKENS,
    cancellation_prompt_words=CANCELLATION_PROMPT_WORDS,
    request_timeout_seconds=ROCM_REQUEST_TIMEOUT_SECONDS,
    max_steady_state_warmup_waves=MAX_STEADY_STATE_WARMUP_WAVES,
    graph_cache_max=ROCM_GRAPH_CACHE_MAX,
    min_stabilization_cycles=MIN_STABILIZATION_CYCLES,
    max_stabilization_cycles=MAX_STABILIZATION_CYCLES,
    required_stable_cycles=REQUIRED_STABLE_CYCLES,
    stabilization_gpu_delta_limit_bytes=STABILIZATION_GPU_DELTA_LIMIT_BYTES,
    stabilization_rss_delta_limit_bytes=STABILIZATION_RSS_DELTA_LIMIT_BYTES,
    active_gpu_peak_growth_limit_bytes=None,
    vulkan_allocation_growth_limit_count=None,
    vulkan_buffer_pool_gb=3.0,
    setup_deadline_seconds=SETUP_DEADLINE_SECONDS,
    host_mem_available_floor_bytes=HOST_MEMORY_AVAILABLE_FLOOR_BYTES,
    host_swap_growth_limit_bytes=HOST_SWAP_GROWTH_LIMIT_BYTES,
    accelerator_telemetry_required=True,
)
ROCM_ENDURANCE_RUNTIME = dataclasses.replace(
    ROCM_RUNTIME,
    variant_id=ROCM_ENDURANCE_VARIANT,
)
VULKAN_RUNTIME = SoakRuntime(
    variant_id=VULKAN_RUNTIME_VARIANT,
    backend="vulkan",
    build_spec=mixed.VULKAN_BUILD_SPEC,
    gpu_memory_scope="server_process",
    gpu_memory_source="linux_proc_drm_fdinfo:vram+gtt+cpu",
    graph_execution_required=False,
    wave_concurrency=VULKAN_QUALIFIED_WAVE_CONCURRENCY,
    prompt_words=VULKAN_QUALIFIED_PROMPT_WORDS,
    prompt_assignment=COHORT_BY_CYCLE,
    max_tokens=16,
    cancel_every_waves=4,
    cancellation_max_tokens=256,
    cancellation_prompt_words=48,
    request_timeout_seconds=600.0,
    max_steady_state_warmup_waves=12,
    graph_cache_max=12,
    min_stabilization_cycles=4,
    max_stabilization_cycles=8,
    required_stable_cycles=2,
    stabilization_gpu_delta_limit_bytes=64 * 1024 * 1024,
    stabilization_rss_delta_limit_bytes=16 * 1024 * 1024,
    active_gpu_peak_growth_limit_bytes=VULKAN_ACTIVE_GPU_PEAK_GROWTH_LIMIT_BYTES,
    vulkan_allocation_growth_limit_count=0,
    vulkan_buffer_pool_gb=VULKAN_QUALIFIED_BUFFER_POOL_GB,
    setup_deadline_seconds=1800.0,
    host_mem_available_floor_bytes=HOST_MEMORY_AVAILABLE_FLOOR_BYTES,
    host_swap_growth_limit_bytes=HOST_SWAP_GROWTH_LIMIT_BYTES,
)
VULKAN_ENDURANCE_RUNTIME = dataclasses.replace(
    VULKAN_RUNTIME,
    variant_id=VULKAN_ENDURANCE_VARIANT,
)
CUDA_ENDURANCE_RUNTIME = SoakRuntime(
    variant_id=CUDA_ENDURANCE_VARIANT,
    backend="cuda",
    build_spec=CUDA_BUILD_SPEC,
    gpu_memory_scope="device_global",
    gpu_memory_source='server_metrics:kiln_gpu_memory_bytes{kind="used"}',
    graph_execution_required=False,
    wave_concurrency=CUDA_QUALIFIED_WAVE_CONCURRENCY,
    prompt_words=CUDA_QUALIFIED_PROMPT_WORDS,
    prompt_assignment=COHORT_BY_CYCLE,
    max_tokens=32,
    cancel_every_waves=4,
    cancellation_max_tokens=512,
    cancellation_prompt_words=48,
    request_timeout_seconds=600.0,
    max_steady_state_warmup_waves=16,
    graph_cache_max=mixed.CUDA_GRAPH_CACHE_ENTRIES,
    min_stabilization_cycles=(
        CUDA_STABILIZATION_MIN_ROTATIONS * len(CUDA_QUALIFIED_PROMPT_WORDS)
    ),
    max_stabilization_cycles=(
        CUDA_STABILIZATION_MAX_ROTATIONS * len(CUDA_QUALIFIED_PROMPT_WORDS)
    ),
    required_stable_cycles=len(CUDA_QUALIFIED_PROMPT_WORDS),
    stabilization_gpu_delta_limit_bytes=64 * 1024 * 1024,
    stabilization_rss_delta_limit_bytes=16 * 1024 * 1024,
    active_gpu_peak_growth_limit_bytes=DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
    vulkan_allocation_growth_limit_count=None,
    vulkan_buffer_pool_gb=0.0,
    setup_deadline_seconds=1800.0,
    host_mem_available_floor_bytes=HOST_MEMORY_AVAILABLE_FLOOR_BYTES,
    host_swap_growth_limit_bytes=HOST_SWAP_GROWTH_LIMIT_BYTES,
    inference_memory_fraction=0.7,
    memory_floor_gb=1.5,
    accelerator_telemetry_enabled=False,
)
RUNTIMES = {
    runtime.variant_id: runtime
    for runtime in (
        ROCM_RUNTIME,
        ROCM_ENDURANCE_RUNTIME,
        VULKAN_RUNTIME,
        VULKAN_ENDURANCE_RUNTIME,
        CUDA_ENDURANCE_RUNTIME,
    )
}


def runtime_for_variant(variant: str) -> SoakRuntime:
    try:
        return RUNTIMES[variant]
    except KeyError as exc:
        raise SoakError(
            f"{VARIANT_ENV} must name one of {sorted(RUNTIMES)}, got {variant!r}"
        ) from exc


def build_phase_deadline(started: float, runtime: SoakRuntime) -> float:
    return started + runtime.build_spec.timeout_seconds


def setup_phase_deadline(started: float, runtime: SoakRuntime) -> float:
    return started + runtime.setup_deadline_seconds


def build_binary_for_soak(
    runtime: SoakRuntime,
) -> tuple[Path, str, float, float, float]:
    build_started = time.monotonic()
    binary, binary_hash, build_seconds = mixed.build_binary(
        build_phase_deadline(build_started, runtime), runtime.build_spec
    )
    setup_started = time.monotonic()
    return (
        binary,
        binary_hash,
        build_seconds,
        setup_started,
        setup_phase_deadline(setup_started, runtime),
    )


def measurement_phase_deadline(
    started: float,
    minimum_duration_seconds: float,
    runtime: SoakRuntime,
) -> float:
    return started + minimum_duration_seconds + runtime.request_timeout_seconds


def qualification_case_timeout_seconds(
    minimum_duration_seconds: float,
    runtime: SoakRuntime,
) -> int:
    return math.ceil(
        runtime.build_spec.timeout_seconds
        + runtime.setup_deadline_seconds
        + minimum_duration_seconds
        + runtime.request_timeout_seconds
        + CASE_TEARDOWN_GRACE_SECONDS
    )


def phase_elapsed_seconds(started: float | None, now: float | None = None) -> float:
    if started is None:
        return 0.0
    finished = time.monotonic() if now is None else now
    return max(0.0, finished - started)

METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    "attributed_itl_outlier_count": ("count", "sum", True),
    "batching_error_count": ("count", "sum", True),
    "batching_max_observed_active_requests": ("requests", "max", False),
    "batching_max_observed_batch_size": ("rows", "max", False),
    "cancellation_confirmed_count": ("count", "sum", False),
    "completion_token_count": ("tokens", "sum", False),
    "output_token_throughput_per_second": ("tokens/s", "rate", False),
    "device_fault_event_count": ("count", "sum", True),
    "external_yield_sync_failure_count": ("count", "sum", True),
    "external_yield_sync_max_ms": ("ms", "max", True),
    "external_yield_sync_slow_count": ("count", "sum", True),
    "graph_capture_failure_count": ("count", "sum", True),
    "graph_capture_success_count": ("count", "sum", False),
    "graph_fallback_count": ("count", "sum", True),
    "graph_retained_count_end": ("graphs", "exact", False),
    "graph_retained_count_start": ("graphs", "exact", False),
    "graph_replay_failure_count": ("count", "sum", True),
    "graph_replay_success_count": ("count", "sum", False),
    "graph_slot_active_count_end": ("slots", "exact", True),
    "graph_slot_count_end": ("slots", "exact", False),
    "graph_slot_count_start": ("slots", "exact", False),
    "graph_slot_create_count": ("count", "sum", True),
    "graph_slot_idle_count_end": ("slots", "exact", False),
    "graph_slot_reuse_count": ("count", "sum", False),
    "gpu_memory_baseline_bytes": ("bytes", "exact", True),
    "gpu_memory_end_bytes": ("bytes", "exact", True),
    "gpu_memory_growth_bytes": ("bytes", "exact", True),
    "gpu_memory_peak_bytes": ("bytes", "max", True),
    "gpu_memory_peak_growth_bytes": ("bytes", "max", True),
    "itl_ms_p50": ("ms", "p50", True),
    "itl_ms_p99": ("ms", "p99", True),
    "itl_ms_p999": ("ms", "p99.9", True),
    "kv_blocks_end": ("blocks", "exact", True),
    "kv_blocks_start": ("blocks", "exact", True),
    "kv_blocks_used_end": ("blocks", "exact", True),
    "kv_unaccounted_blocks_end": ("blocks", "exact", True),
    "non_finite_response_count": ("count", "sum", True),
    "prefix_cache_active_leases_end": ("leases", "exact", True),
    "prefix_cache_baseline_cached_entries": ("entries", "exact", False),
    "prefix_cache_baseline_state_bytes": ("bytes", "exact", True),
    "prefix_cache_cached_blocks_end": ("blocks", "exact", False),
    "prefix_cache_cached_entries_end": ("entries", "exact", False),
    "prefix_cache_hit_blocks": ("blocks", "sum", False),
    "prefix_cache_hit_tokens": ("tokens", "sum", False),
    "prefix_cache_lookup_hit_count": ("count", "sum", False),
    "prefix_cache_lookup_miss_count": ("count", "sum", True),
    "prefix_cache_pending_release_entries_end": ("entries", "exact", True),
    "prefix_cache_state_bytes_end": ("bytes", "exact", True),
    "prompt_tokens_max": ("tokens", "max", False),
    "prompt_tokens_min": ("tokens", "min", False),
    "request_failure_count": ("count", "sum", True),
    "request_count": ("count", "sum", False),
    "request_worker_residue_count": ("count", "max", True),
    "rss_baseline_bytes": ("bytes", "exact", True),
    "rss_end_bytes": ("bytes", "exact", True),
    "rss_growth_bytes": ("bytes", "exact", True),
    "rss_peak_bytes": ("bytes", "max", True),
    "measurement_final_snapshot_complete": ("bool", "exact", False),
    "shutdown_forced_count": ("count", "sum", True),
    "shutdown_nonzero_count": ("count", "sum", True),
    "snapshot_residue_count": ("count", "sum", True),
    "soak_duration_seconds": ("s", "sum", False),
    "steady_state_warmup_request_count": ("count", "sum", False),
    "steady_state_warmup_wave_count": ("count", "sum", False),
    "stabilization_cancellation_count": ("count", "sum", False),
    "stabilization_cycle_count": ("count", "sum", False),
    "stabilization_final_gpu_delta_bytes": ("bytes", "max", True),
    "stabilization_final_rss_delta_bytes": ("bytes", "max", True),
    "stabilization_max_gpu_delta_bytes": ("bytes", "max", True),
    "stabilization_max_rss_delta_bytes": ("bytes", "max", True),
    "stabilization_rss_growth_bytes": ("bytes", "exact", True),
    "stabilization_request_count": ("count", "sum", False),
    "stabilization_stable_cycle_count": ("count", "sum", False),
    "ttft_ms_p50": ("ms", "p50", True),
    "ttft_ms_p99": ("ms", "p99", True),
    "ttft_ms_p999": ("ms", "p99.9", True),
    "unexplained_itl_outlier_count": ("count", "sum", True),
    "wave_count": ("count", "sum", False),
    "zero_token_response_count": ("count", "sum", True),
}
METRIC_DEFINITIONS["latency_phase_metadata_missing_count"] = (
    "count",
    "sum",
    True,
)
for _phase_name in mixed.LATENCY_PHASE_NAMES:
    METRIC_DEFINITIONS[f"latency_phase_{_phase_name}_ms_total"] = (
        "ms",
        "sum",
        True,
    )
    METRIC_DEFINITIONS[f"latency_phase_{_phase_name}_request_count"] = (
        "count",
        "sum",
        False,
    )
HOST_SAFETY_METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    "host_mem_available_end_bytes": ("bytes", "exact", False),
    "host_mem_available_min_bytes": ("bytes", "min", False),
    "host_mem_available_start_bytes": ("bytes", "exact", False),
    "host_memory_guard_trip_count": ("count", "sum", True),
    "host_swap_growth_bytes": ("bytes", "max", True),
    "host_swap_used_end_bytes": ("bytes", "exact", True),
    "host_swap_used_peak_bytes": ("bytes", "max", True),
    "host_swap_used_start_bytes": ("bytes", "exact", True),
}
ACCELERATOR_TELEMETRY_METRIC_DEFINITIONS: dict[
    str, tuple[str, str, bool]
] = {
    "accelerator_gpu_busy_percent_p50": ("percent", "p50", False),
    "accelerator_gpu_busy_percent_peak": ("percent", "max", False),
    "accelerator_power_active_p50_microwatts": ("microwatts", "p50", True),
    "accelerator_power_active_peak_microwatts": ("microwatts", "max", True),
    "accelerator_sclk_active_below_half_max_count": ("count", "sum", True),
    "accelerator_sclk_active_max_hz": ("hz", "max", False),
    "accelerator_sclk_active_min_hz": ("hz", "min", False),
    "accelerator_sclk_active_p50_hz": ("hz", "p50", False),
    "accelerator_sclk_advertised_max_hz": ("hz", "exact", False),
    "accelerator_telemetry_active_sample_count": ("count", "sum", False),
    "accelerator_telemetry_available": ("bool", "exact", False),
    "accelerator_telemetry_error_count": ("count", "sum", True),
    "accelerator_telemetry_sample_count": ("count", "sum", False),
}
ROCM_METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    **METRIC_DEFINITIONS,
    **HOST_SAFETY_METRIC_DEFINITIONS,
    **ACCELERATOR_TELEMETRY_METRIC_DEFINITIONS,
}
VULKAN_METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    **METRIC_DEFINITIONS,
    **HOST_SAFETY_METRIC_DEFINITIONS,
    **ACCELERATOR_TELEMETRY_METRIC_DEFINITIONS,
    "prefix_cache_enabled": ("bool", "exact", True),
    "resident_prefill_enabled": ("bool", "exact", False),
    "batched_state_cache_active_leases_end": ("leases", "exact", True),
    "batched_state_cache_capacity_rows_end": ("rows", "exact", False),
    "batched_state_cache_completed_row_eviction_count": ("count", "sum", True),
    "batched_state_cache_completed_row_preservation_count": (
        "count",
        "sum",
        False,
    ),
    "batched_state_cache_entry_present_end": ("entries", "exact", False),
    "batched_state_cache_exact_reuse_count": ("count", "sum", False),
    "batched_state_cache_explicit_invalidation_count": ("count", "sum", True),
    "batched_state_cache_explicit_invalidation_eviction_count": (
        "count",
        "sum",
        True,
    ),
    "batched_state_cache_fresh_assembly_count": ("count", "sum", True),
    "batched_state_cache_lease_drop_eviction_count": ("count", "sum", True),
    "batched_state_cache_logical_rows_end": ("rows", "exact", False),
    "batched_state_cache_max_active_leases": ("leases", "max", True),
    "batched_state_cache_park_count": ("count", "sum", False),
    "batched_state_cache_park_replacement_eviction_count": (
        "count",
        "sum",
        True,
    ),
    "batched_state_cache_rejected_insufficient_capacity_count": (
        "count",
        "sum",
        True,
    ),
    "batched_state_cache_rejected_missing_row_ids_count": ("count", "sum", True),
    "batched_state_cache_rejected_nonresident_cache_count": ("count", "sum", True),
    "batched_state_cache_rejected_nonresident_rows_count": ("count", "sum", True),
    "batched_state_cache_resident_prefix_snapshot_suppression_count": (
        "count",
        "sum",
        False,
    ),
    "batched_state_cache_resident_capacity_reuse_count": ("count", "sum", False),
    "batched_state_cache_resident_end": ("count", "exact", False),
    "batched_state_cache_resident_prefix_view_count": ("count", "sum", False),
    "batched_state_cache_resident_refresh_count": ("count", "sum", False),
    "batched_state_cache_take_hit_count": ("count", "sum", False),
    "batched_state_cache_take_miss_count": ("count", "sum", True),
    "batched_state_cache_take_miss_while_leased_count": ("count", "sum", True),
    "resident_recurrent_state_allocation_bytes_end": ("bytes", "exact", True),
    "resident_recurrent_state_buffer_bytes_end": ("bytes", "exact", True),
    "resident_recurrent_state_entries_end": ("entries", "exact", True),
    "resident_prefill_active_rows_end": ("rows", "exact", True),
    "resident_prefill_attempt_count": ("count", "sum", False),
    "resident_prefill_completed_row_count": ("rows", "sum", False),
    "resident_prefill_forward_count": ("count", "sum", False),
    "resident_prefill_initial_decline_count": ("count", "sum", True),
    "resident_prefill_max_batch_size": ("rows", "max", False),
    "resident_prefill_route_failure_count": ("count", "sum", True),
    "resident_prefill_row_count": ("rows", "sum", False),
    "stabilization_resident_prefill_active_rows_end": ("rows", "exact", True),
    "stabilization_resident_prefill_attempt_count": ("count", "sum", False),
    "stabilization_resident_prefill_completed_row_count": (
        "rows",
        "sum",
        False,
    ),
    "stabilization_resident_prefill_enabled": ("bool", "exact", False),
    "stabilization_resident_prefill_forward_count": ("count", "sum", False),
    "stabilization_resident_prefill_initial_decline_count": (
        "count",
        "sum",
        True,
    ),
    "stabilization_resident_prefill_max_batch_size": ("rows", "max", False),
    "stabilization_resident_prefill_route_failure_count": ("count", "sum", True),
    "stabilization_resident_prefill_row_count": ("rows", "sum", False),
    **{
        f"vulkan_process_mapping_{category}_rss_growth_bytes": (
            "bytes",
            "exact",
            True,
        )
        for category in PROCESS_MEMORY_MAPPING_CATEGORIES
    },
    "vulkan_process_smaps_anonymous_growth_bytes": ("bytes", "exact", True),
    "vulkan_process_smaps_anonymous_huge_pages_growth_bytes": (
        "bytes",
        "exact",
        True,
    ),
    "vulkan_process_smaps_private_dirty_growth_bytes": ("bytes", "exact", True),
    "vulkan_process_smaps_rss_end_bytes": ("bytes", "exact", True),
    "vulkan_process_smaps_rss_start_bytes": ("bytes", "exact", True),
    "vulkan_process_smaps_swap_growth_bytes": ("bytes", "exact", True),
    "vulkan_buffer_allocated_bytes": ("bytes", "sum", True),
    "vulkan_buffer_allocation_count": ("count", "sum", True),
    "vulkan_buffer_free_count": ("count", "sum", False),
    "vulkan_buffer_freed_bytes": ("bytes", "sum", False),
    "vulkan_buffer_live_bytes_end": ("bytes", "exact", True),
    "vulkan_buffer_live_bytes_growth": ("bytes", "exact", True),
    "vulkan_buffer_live_bytes_start": ("bytes", "exact", True),
    "vulkan_buffer_peak_live_bytes": ("bytes", "max", True),
    "vulkan_buffer_stabilization_growth_cycle_count": ("count", "sum", True),
    "vulkan_buffer_pool_cache_hit_count": ("count", "sum", False),
    "vulkan_buffer_pool_cache_miss_count": ("count", "sum", True),
    "vulkan_buffer_pool_device_local_cache_miss_count": ("count", "sum", True),
    "vulkan_buffer_pool_evicted_bytes": ("bytes", "sum", True),
    "vulkan_buffer_pool_eviction_count": ("count", "sum", True),
    "vulkan_buffer_pool_host_visible_cache_miss_count": ("count", "sum", True),
    "vulkan_buffer_pool_limit_bytes": ("bytes", "exact", True),
    "vulkan_buffer_pool_retained_bytes_end": ("bytes", "exact", True),
    "vulkan_buffer_pool_retained_bytes_growth": ("bytes", "exact", True),
    "vulkan_buffer_pool_retained_bytes_start": ("bytes", "exact", True),
    "vulkan_buffer_pool_uncached_allocated_bytes": ("bytes", "sum", True),
    "vulkan_buffer_pool_uncached_allocation_count": ("count", "sum", True),
}
CUDA_METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    **METRIC_DEFINITIONS,
    **HOST_SAFETY_METRIC_DEFINITIONS,
}


class SoakError(RuntimeError):
    pass


class AcceleratorTelemetryUnavailable(SoakError):
    pass


@dataclasses.dataclass
class RequestWorkerEvidence:
    peak_residue_count: int = 0

    def observe(self, residue_count: int) -> None:
        self.peak_residue_count = max(self.peak_residue_count, residue_count)



def effective_config(
    minimum_duration_seconds: float,
    memory_growth_limit_bytes: int,
    runtime: SoakRuntime = ROCM_RUNTIME,
) -> dict[str, Any]:
    base = mixed.VARIANT_CONFIGS[runtime.variant_id]
    if runtime.backend == "rocm":
        expected_graph_cache_max = (
            base["server"]["max_active_requests"]
            * ROCM_GRAPH_ACTIVE_OWNER_FLOOR
            + ROCM_GRAPH_TRANSITION_HEADROOM_ENTRIES
        )
        if runtime.graph_cache_max != expected_graph_cache_max:
            raise SoakError(
                "ROCm graph cache must reserve one protected geometry for each "
                "declared active request plus any explicit transition headroom: "
                f"{runtime.graph_cache_max} != {expected_graph_cache_max}"
            )
    effective = {
        "build": base["build"],
        "runtime": base["runtime"],
        "server": base["server"],
        "soak": {
            "cancellation_after_semantic_deltas": mixed.CANCELLATION_AFTER_DELTAS,
            "cancellation_max_tokens": runtime.cancellation_max_tokens,
            "cancellation_prompt_words": runtime.cancellation_prompt_words,
            "cancel_every_waves": runtime.cancel_every_waves,
            "gpu_memory_scope": runtime.gpu_memory_scope,
            "gpu_memory_source": runtime.gpu_memory_source,
            "active_gpu_peak_growth_limit_bytes": (
                runtime.active_gpu_peak_growth_limit_bytes
                if runtime.active_gpu_peak_growth_limit_bytes is not None
                else memory_growth_limit_bytes
            ),
            "max_tokens": runtime.max_tokens,
            "rocm_graph_cache_entries": runtime.graph_cache_max,
            "memory_growth_limit_bytes": memory_growth_limit_bytes,
            "minimum_duration_seconds": minimum_duration_seconds,
            "outlier_absolute_ms": int(mixed.OUTLIER_ABSOLUTE_MS),
            "outlier_history_size": mixed.OUTLIER_HISTORY_SIZE,
            "outlier_multiplier": int(mixed.OUTLIER_MULTIPLIER),
            "prompt_assignment": runtime.prompt_assignment,
            "prompt_identity": (
                "fixed_by_cycle_cohort_measured_unique_by_epoch_warmup"
                if runtime.prompt_assignment == COHORT_BY_CYCLE
                else "fixed_by_slot_measured_unique_by_epoch_warmup"
            ),
            "prompt_words": {
                f"slot_{index}": words
                for index, words in enumerate(runtime.prompt_words)
            },
            "response_oracle": mixed.RESPONSE_ORACLE,
            "response_oracle_integer_width": mixed.RESPONSE_ORACLE_INTEGER_WIDTH,
            "request_ignore_eos": True,
            "request_worker_cleanup_timeout_seconds": (
                REQUEST_WORKER_CLEANUP_TIMEOUT_SECONDS
            ),
            "deadline_policy": "independent_build_setup_and_measurement",
            "measurement_deadline_seconds": int(
                minimum_duration_seconds + runtime.request_timeout_seconds
            ),
            "qualification_case_timeout_seconds": qualification_case_timeout_seconds(
                minimum_duration_seconds, runtime
            ),
            "setup_deadline_seconds": int(runtime.setup_deadline_seconds),
            "teardown_grace_seconds": int(CASE_TEARDOWN_GRACE_SECONDS),
            "stabilization_gpu_delta_limit_bytes": (
                runtime.stabilization_gpu_delta_limit_bytes
            ),
            "stabilization_memory_boundary": (
                "process_drm_and_owned_buffers"
                if runtime.backend == "vulkan"
                else "gpu_and_rss"
            ),
            "stabilization_max_cycles": runtime.max_stabilization_cycles,
            "stabilization_min_cycles": runtime.min_stabilization_cycles,
            "stabilization_required_stable_cycles": runtime.required_stable_cycles,
            "stabilization_rss_delta_limit_bytes": (
                runtime.stabilization_rss_delta_limit_bytes
                if runtime.backend != "vulkan"
                else None
            ),
            "stabilization_rss_growth_limit_bytes": (
                memory_growth_limit_bytes if runtime.backend == "vulkan" else None
            ),
            "steady_state_warmup_max_waves": runtime.max_steady_state_warmup_waves,
            "vulkan_buffer_pool_gb": runtime.vulkan_buffer_pool_gb,
            "wave_concurrency": {
                f"wave_{index}": concurrency
                for index, concurrency in enumerate(runtime.wave_concurrency)
            },
        },
    }
    if "batching" in base:
        effective["batching"] = base["batching"]
    if runtime.backend == "cuda":
        effective["soak"]["gpu_memory_baseline_mode"] = (
            "stabilization_envelope_high_water"
        )
    if "model" in base:
        effective["model"] = base["model"]
    if runtime.inference_memory_fraction is not None:
        effective["memory"] = {
            "floor_gb": runtime.memory_floor_gb,
            "inference_memory_fraction": runtime.inference_memory_fraction,
        }
    if runtime.accelerator_telemetry_enabled:
        effective["soak"]["accelerator_telemetry"] = {
            "active_busy_floor_percent": (
                ACCELERATOR_TELEMETRY_ACTIVE_BUSY_FLOOR_PERCENT
            ),
            "amd_gpu_vendor_id": AMD_GPU_VENDOR_ID,
            "device_selector": "exactly_one_amd_drm_device",
            "mode": (
                "required"
                if runtime.accelerator_telemetry_required
                else "if_available"
            ),
            "poll_interval_ms": int(HOST_GUARD_POLL_INTERVAL_SECONDS * 1000),
            "sources": {
                "busy": "drm_device/gpu_busy_percent",
                "power": "amdgpu_hwmon/power_PPT_average",
                "sclk": "amdgpu_hwmon/freq_sclk_input",
                "sclk_advertised_max": "drm_device/pp_dpm_sclk",
            },
        }
    else:
        effective["soak"]["accelerator_telemetry"] = {"mode": "disabled"}
    if runtime.backend == "rocm":
        effective["soak"]["rocm_graph_admission_policy"] = (
            ROCM_GRAPH_ADMISSION_POLICY
        )
        effective["soak"]["rocm_graph_active_owner_floor"] = (
            ROCM_GRAPH_ACTIVE_OWNER_FLOOR
        )
        effective["soak"]["rocm_graph_transition_headroom_entries"] = (
            ROCM_GRAPH_TRANSITION_HEADROOM_ENTRIES
        )
    if runtime.host_mem_available_floor_bytes is not None:
        effective["soak"]["host_mem_available_floor_bytes"] = (
            runtime.host_mem_available_floor_bytes
        )
        effective["soak"]["host_swap_growth_limit_bytes"] = (
            runtime.host_swap_growth_limit_bytes
        )
        effective["soak"]["host_memory_poll_interval_ms"] = int(
            HOST_GUARD_POLL_INTERVAL_SECONDS * 1000
        )
    if runtime.vulkan_allocation_growth_limit_count is not None:
        effective["soak"]["vulkan_allocation_growth_limit_count"] = (
            runtime.vulkan_allocation_growth_limit_count
        )
    return effective


@dataclasses.dataclass(frozen=True)
class ProcessMemorySnapshot:
    rss_bytes: int
    rss_anon_bytes: int
    rss_file_bytes: int
    rss_shmem_bytes: int
    swap_bytes: int


@dataclasses.dataclass(frozen=True)
class ProcessMemoryMappingUsage:
    identity: str
    category: str
    pathname: str
    size_bytes: int
    rss_bytes: int
    pss_bytes: int
    anonymous_bytes: int
    anonymous_huge_pages_bytes: int
    private_dirty_bytes: int
    swap_bytes: int


@dataclasses.dataclass(frozen=True)
class ProcessMemoryMappingSnapshot:
    mappings: tuple[ProcessMemoryMappingUsage, ...]


def parse_memory_kib(path: Path, name: str, raw: str, unit: str) -> int:
    fields = raw.split()
    if len(fields) != 2 or fields[1] != unit:
        raise SoakError(f"{path} has an invalid {name} value: {raw!r}")
    try:
        value = int(fields[0]) * 1024
    except ValueError as exc:
        raise SoakError(f"{path} has an invalid {name} value: {raw!r}") from exc
    if value < 0:
        raise SoakError(f"{path} has a negative {name} value")
    return value


def process_memory_snapshot(
    pid: int, proc_root: Path = Path("/proc")
) -> ProcessMemorySnapshot:
    status = proc_root / str(pid) / "status"
    names = {"VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap"}
    values: dict[str, int] = {}
    for line in status.read_text(encoding="utf-8").splitlines():
        name, separator, raw = line.partition(":")
        if not separator or name not in names:
            continue
        values[name] = parse_memory_kib(status, name, raw, "kB")
    missing = sorted(names - set(values))
    if missing:
        raise SoakError(f"{status} omitted required fields: {missing}")
    return ProcessMemorySnapshot(
        rss_bytes=values["VmRSS"],
        rss_anon_bytes=values["RssAnon"],
        rss_file_bytes=values["RssFile"],
        rss_shmem_bytes=values["RssShmem"],
        swap_bytes=values["VmSwap"],
    )


def process_memory_mapping_category(pathname: str) -> str:
    if pathname == "[heap]":
        return "heap"
    if pathname == "[stack]" or pathname.startswith("[stack:"):
        return "stack"
    if not pathname or pathname.startswith("[anon:"):
        return "anonymous"
    if (
        pathname.startswith("/dev/shm/")
        or pathname.startswith("/memfd:")
        or pathname.startswith("memfd:")
        or pathname.startswith("/SYSV")
        or pathname.startswith("[anon_shmem:")
    ):
        return "shared_memory"
    if pathname.startswith("/dev/"):
        return "device"
    if pathname.startswith("["):
        return "kernel"
    return "file"


def process_memory_mapping_snapshot(
    pid: int, proc_root: Path = Path("/proc")
) -> ProcessMemoryMappingSnapshot:
    smaps = proc_root / str(pid) / "smaps"
    mappings: list[ProcessMemoryMappingUsage] = []
    address_range: str | None = None
    pathname = ""
    values: dict[str, int] = {}

    def finish_mapping() -> None:
        if address_range is None:
            return
        missing = sorted(set(PROCESS_MEMORY_MAPPING_FIELDS) - set(values))
        if missing:
            raise SoakError(
                f"{smaps} mapping {address_range} omitted required fields: {missing}"
            )
        for field in (
            "Rss",
            "Pss",
            "Anonymous",
            "AnonHugePages",
            "Private_Dirty",
        ):
            if values[field] > values["Size"]:
                raise SoakError(
                    f"{smaps} mapping {address_range} has {field}="
                    f"{values[field]} above Size={values['Size']}"
                )
        if values["Anonymous"] > values["Rss"]:
            raise SoakError(
                f"{smaps} mapping {address_range} has Anonymous="
                f"{values['Anonymous']} above Rss={values['Rss']}"
            )
        if values["Private_Dirty"] > values["Rss"]:
            raise SoakError(
                f"{smaps} mapping {address_range} has Private_Dirty="
                f"{values['Private_Dirty']} above Rss={values['Rss']}"
            )
        if values["AnonHugePages"] > values["Anonymous"]:
            raise SoakError(
                f"{smaps} mapping {address_range} has AnonHugePages="
                f"{values['AnonHugePages']} above Anonymous={values['Anonymous']}"
            )
        category = process_memory_mapping_category(pathname)
        label = pathname or "[anonymous]"
        mappings.append(
            ProcessMemoryMappingUsage(
                identity=f"{label}@{address_range}",
                category=category,
                pathname=pathname,
                size_bytes=values["Size"],
                rss_bytes=values["Rss"],
                pss_bytes=values["Pss"],
                anonymous_bytes=values["Anonymous"],
                anonymous_huge_pages_bytes=values["AnonHugePages"],
                private_dirty_bytes=values["Private_Dirty"],
                swap_bytes=values["Swap"],
            )
        )

    with smaps.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            header = SMAPS_HEADER.fullmatch(line)
            if header is not None:
                finish_mapping()
                address_range = header.group("range")
                pathname = header.group("pathname") or ""
                values = {}
                continue
            if address_range is None:
                raise SoakError(f"{smaps} has content before its first mapping header")
            name, separator, raw = line.partition(":")
            if separator and name in PROCESS_MEMORY_MAPPING_FIELDS:
                if name in values:
                    raise SoakError(
                        f"{smaps} mapping {address_range} repeats field {name}"
                    )
                values[name] = parse_memory_kib(smaps, name, raw, "kB")
    finish_mapping()
    if not mappings:
        raise SoakError(f"{smaps} contains no memory mappings")
    return ProcessMemoryMappingSnapshot(mappings=tuple(mappings))


def process_memory_mapping_totals(
    snapshot: ProcessMemoryMappingSnapshot,
) -> dict[str, int]:
    totals = {
        "rss_bytes": 0,
        "pss_bytes": 0,
        "anonymous_bytes": 0,
        "anonymous_huge_pages_bytes": 0,
        "private_dirty_bytes": 0,
        "swap_bytes": 0,
        **{f"{category}_rss_bytes": 0 for category in PROCESS_MEMORY_MAPPING_CATEGORIES},
    }
    for mapping in snapshot.mappings:
        if mapping.category not in PROCESS_MEMORY_MAPPING_CATEGORIES:
            raise SoakError(f"unknown process-memory mapping category {mapping.category!r}")
        totals["rss_bytes"] += mapping.rss_bytes
        totals["pss_bytes"] += mapping.pss_bytes
        totals["anonymous_bytes"] += mapping.anonymous_bytes
        totals["anonymous_huge_pages_bytes"] += mapping.anonymous_huge_pages_bytes
        totals["private_dirty_bytes"] += mapping.private_dirty_bytes
        totals["swap_bytes"] += mapping.swap_bytes
        totals[f"{mapping.category}_rss_bytes"] += mapping.rss_bytes
    return totals


def process_memory_mapping_trace(
    before: ProcessMemoryMappingSnapshot,
    after: ProcessMemoryMappingSnapshot,
    *,
    top_limit: int = 8,
) -> dict[str, Any]:
    if top_limit < 1:
        raise SoakError("process-memory mapping trace top_limit must be positive")
    before_totals = process_memory_mapping_totals(before)
    after_totals = process_memory_mapping_totals(after)
    before_by_identity = {mapping.identity: mapping for mapping in before.mappings}
    top_growth: list[dict[str, int | str]] = []
    for mapping in after.mappings:
        prior = before_by_identity.get(mapping.identity)
        rss_delta = mapping.rss_bytes - (prior.rss_bytes if prior is not None else 0)
        if rss_delta <= 0:
            continue
        top_growth.append(
            {
                "anonymous_bytes": mapping.anonymous_bytes,
                "anonymous_huge_pages_bytes": mapping.anonymous_huge_pages_bytes,
                "category": mapping.category,
                "identity": mapping.identity,
                "private_dirty_bytes": mapping.private_dirty_bytes,
                "rss_bytes": mapping.rss_bytes,
                "rss_delta_bytes": rss_delta,
                "size_bytes": mapping.size_bytes,
            }
        )
    top_growth.sort(key=lambda item: (-int(item["rss_delta_bytes"]), str(item["identity"])))
    return {
        "smaps_anonymous_delta_bytes": (
            after_totals["anonymous_bytes"] - before_totals["anonymous_bytes"]
        ),
        "smaps_anonymous_huge_pages_delta_bytes": (
            after_totals["anonymous_huge_pages_bytes"]
            - before_totals["anonymous_huge_pages_bytes"]
        ),
        "smaps_private_dirty_delta_bytes": (
            after_totals["private_dirty_bytes"]
            - before_totals["private_dirty_bytes"]
        ),
        "smaps_pss_delta_bytes": after_totals["pss_bytes"] - before_totals["pss_bytes"],
        "smaps_rss_bytes_by_category": {
            category: after_totals[f"{category}_rss_bytes"]
            for category in PROCESS_MEMORY_MAPPING_CATEGORIES
        },
        "smaps_rss_delta_bytes_by_category": {
            category: (
                after_totals[f"{category}_rss_bytes"]
                - before_totals[f"{category}_rss_bytes"]
            )
            for category in PROCESS_MEMORY_MAPPING_CATEGORIES
        },
        "smaps_swap_delta_bytes": after_totals["swap_bytes"] - before_totals["swap_bytes"],
        "smaps_top_rss_growth": top_growth[:top_limit],
    }


def process_memory_mapping_metric_values(
    before: ProcessMemoryMappingSnapshot,
    after: ProcessMemoryMappingSnapshot,
) -> dict[str, int]:
    before_totals = process_memory_mapping_totals(before)
    after_totals = process_memory_mapping_totals(after)
    values = {
        "vulkan_process_smaps_anonymous_huge_pages_growth_bytes": max(
            0,
            after_totals["anonymous_huge_pages_bytes"]
            - before_totals["anonymous_huge_pages_bytes"],
        ),
        "vulkan_process_smaps_anonymous_growth_bytes": max(
            0, after_totals["anonymous_bytes"] - before_totals["anonymous_bytes"]
        ),
        "vulkan_process_smaps_private_dirty_growth_bytes": max(
            0,
            after_totals["private_dirty_bytes"]
            - before_totals["private_dirty_bytes"],
        ),
        "vulkan_process_smaps_rss_end_bytes": after_totals["rss_bytes"],
        "vulkan_process_smaps_rss_start_bytes": before_totals["rss_bytes"],
        "vulkan_process_smaps_swap_growth_bytes": max(
            0, after_totals["swap_bytes"] - before_totals["swap_bytes"]
        ),
    }
    values.update(
        {
            f"vulkan_process_mapping_{category}_rss_growth_bytes": max(
                0,
                after_totals[f"{category}_rss_bytes"]
                - before_totals[f"{category}_rss_bytes"],
            )
            for category in PROCESS_MEMORY_MAPPING_CATEGORIES
        }
    )
    return values


def host_memory_snapshot() -> tuple[int, int]:
    meminfo = Path("/proc/meminfo")
    fields: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        name, separator, raw = line.partition(":")
        if not separator or name not in {"MemAvailable", "SwapTotal", "SwapFree"}:
            continue
        fields[name] = parse_memory_kib(meminfo, name, raw, "kB")
    missing = sorted({"MemAvailable", "SwapTotal", "SwapFree"} - set(fields))
    if missing:
        raise SoakError(f"/proc/meminfo omitted required fields: {missing}")
    if fields["SwapFree"] > fields["SwapTotal"]:
        raise SoakError("/proc/meminfo reports more free swap than total swap")
    return fields["MemAvailable"], fields["SwapTotal"] - fields["SwapFree"]


class HostMemoryGuard:
    def __init__(
        self,
        process: subprocess.Popen[str],
        available_floor_bytes: int,
    ) -> None:
        self.process = process
        self.available_floor_bytes = available_floor_bytes
        self.stop = threading.Event()
        self.samples: list[tuple[int, int]] = []
        self.errors: list[str] = []
        self.trip_reason: str | None = None
        self.thread = threading.Thread(
            target=self._run, name="qualification-host-memory-guard"
        )
        self._started = False
        self._closed = False

    def start(self) -> None:
        self._sample()
        if self.trip_reason is not None:
            return
        self.thread.start()
        self._started = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.trip_reason is None:
            self._sample()
        self.stop.set()
        if not self._started:
            return
        self.thread.join(timeout=10.0)
        if self.thread.is_alive() and len(self.errors) < 8:
            self.errors.append("host-memory-guard thread did not stop within 10 seconds")

    def metric_values(self) -> dict[str, int]:
        if not self.samples:
            return {
                "host_mem_available_end_bytes": 0,
                "host_mem_available_min_bytes": 0,
                "host_mem_available_start_bytes": 0,
                "host_memory_guard_trip_count": int(self.trip_reason is not None),
                "host_swap_growth_bytes": 0,
                "host_swap_used_end_bytes": 0,
                "host_swap_used_peak_bytes": 0,
                "host_swap_used_start_bytes": 0,
            }
        available = [sample[0] for sample in self.samples]
        swap = [sample[1] for sample in self.samples]
        return {
            "host_mem_available_end_bytes": available[-1],
            "host_mem_available_min_bytes": min(available),
            "host_mem_available_start_bytes": available[0],
            "host_memory_guard_trip_count": int(self.trip_reason is not None),
            "host_swap_growth_bytes": max(0, max(swap) - swap[0]),
            "host_swap_used_end_bytes": swap[-1],
            "host_swap_used_peak_bytes": max(swap),
            "host_swap_used_start_bytes": swap[0],
        }

    def _trip(self, reason: str) -> None:
        if self.trip_reason is not None:
            return
        self.trip_reason = reason
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def _sample(self) -> None:
        try:
            available, swap = host_memory_snapshot()
            self.samples.append((available, swap))
            if available < self.available_floor_bytes:
                self._trip(
                    f"host MemAvailable fell to {available} bytes below the "
                    f"{self.available_floor_bytes}-byte safety floor"
                )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if len(self.errors) < 8:
                self.errors.append(message)
            self._trip(f"host memory guard failed closed: {message}")

    def _run(self) -> None:
        while not self.stop.wait(HOST_GUARD_POLL_INTERVAL_SECONDS):
            self._sample()
            if self.trip_reason is not None:
                return


@dataclasses.dataclass(frozen=True)
class AcceleratorTelemetryPaths:
    drm_device: Path
    busy_percent: Path
    sclk_hz: Path
    power_microwatts: Path
    advertised_max_sclk_hz: int


@dataclasses.dataclass(frozen=True)
class AcceleratorTelemetrySample:
    observed: float
    busy_percent: int
    sclk_hz: int
    power_microwatts: int


def _read_bounded_decimal(
    path: Path,
    *,
    minimum: int,
    maximum: int,
    label: str,
) -> int:
    raw = path.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[+-]?\d+", raw) is None:
        raise SoakError(f"{path} contains a non-integer {label} value {raw!r}")
    value = int(raw)
    if value < minimum or value > maximum:
        raise SoakError(
            f"{path} {label} value {value} is outside {minimum}..={maximum}"
        )
    return value


def _resolve_labeled_hwmon_input(
    hwmon: Path,
    *,
    prefix: str,
    label: str,
    value_suffix: str = "input",
) -> Path:
    matches: list[Path] = []
    for label_path in sorted(hwmon.glob(f"{prefix}*_label")):
        if label_path.read_text(encoding="utf-8").strip() != label:
            continue
        input_path = label_path.with_name(
            label_path.name.removesuffix("_label") + f"_{value_suffix}"
        )
        if not input_path.is_file():
            raise SoakError(f"{label_path} has no matching input")
        matches.append(input_path)
    if len(matches) != 1:
        raise SoakError(
            f"amdgpu hwmon {prefix}/{label} resolved to {len(matches)} inputs, "
            "expected exactly one"
        )
    return matches[0]


def _advertised_max_sclk_hz(path: Path) -> int:
    frequencies: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"\s*\d+:\s*(\d+)Mhz(?:\s+\*)?\s*", line)
        if match is not None:
            frequencies.append(int(match.group(1)) * 1_000_000)
    if not frequencies:
        raise SoakError(f"{path} contains no parseable SCLK performance levels")
    return max(frequencies)


def resolve_amd_accelerator_telemetry_paths(
    drm_root: Path = Path("/sys/class/drm"),
) -> AcceleratorTelemetryPaths:
    devices: list[Path] = []
    for card in sorted(drm_root.glob("card[0-9]*")):
        if re.fullmatch(r"card\d+", card.name) is None:
            continue
        device = card / "device"
        vendor_path = device / "vendor"
        if not vendor_path.is_file():
            continue
        if (
            vendor_path.read_text(encoding="utf-8").strip().lower()
            == AMD_GPU_VENDOR_ID
        ):
            devices.append(device)
    if not devices:
        raise AcceleratorTelemetryUnavailable(
            "AMD accelerator telemetry resolved 0 DRM devices"
        )
    if len(devices) != 1:
        raise SoakError(
            f"AMD accelerator telemetry resolved {len(devices)} DRM devices, "
            "expected exactly one"
        )
    device = devices[0]
    hwmons = [
        path
        for path in sorted((device / "hwmon").glob("hwmon*"))
        if (path / "name").is_file()
        and (path / "name").read_text(encoding="utf-8").strip() == "amdgpu"
    ]
    if len(hwmons) != 1:
        raise SoakError(
            f"AMD accelerator telemetry resolved {len(hwmons)} amdgpu hwmon devices, "
            "expected exactly one"
        )
    hwmon = hwmons[0]
    busy_percent = device / "gpu_busy_percent"
    advertised_sclk = device / "pp_dpm_sclk"
    for path in (busy_percent, advertised_sclk):
        if not path.is_file():
            raise SoakError(f"AMD accelerator telemetry source is missing: {path}")
    return AcceleratorTelemetryPaths(
        drm_device=device,
        busy_percent=busy_percent,
        sclk_hz=_resolve_labeled_hwmon_input(hwmon, prefix="freq", label="sclk"),
        power_microwatts=_resolve_labeled_hwmon_input(
            hwmon, prefix="power", label="PPT", value_suffix="average"
        ),
        advertised_max_sclk_hz=_advertised_max_sclk_hz(advertised_sclk),
    )


class AcceleratorTelemetrySampler:
    def __init__(
        self,
        *,
        enabled: bool = True,
        required: bool,
        drm_root: Path = Path("/sys/class/drm"),
    ) -> None:
        self.enabled = enabled
        self.required = required
        self.drm_root = drm_root
        self.paths: AcceleratorTelemetryPaths | None = None
        self.samples: list[AcceleratorTelemetrySample] = []
        self.errors: list[str] = []
        self.unavailable_reason: str | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="qualification-accelerator-telemetry",
            daemon=True,
        )
        self._started = False
        self._closed = False

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            self.paths = resolve_amd_accelerator_telemetry_paths(self.drm_root)
            self._sample()
            if self.errors:
                self.unavailable_reason = self.errors[-1]
                mixed.trace(
                    "accelerator_telemetry_unavailable",
                    required=self.required,
                    reason=self.unavailable_reason,
                )
                return
        except AcceleratorTelemetryUnavailable as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.unavailable_reason = message
            if self.required:
                self.errors.append(message)
            mixed.trace(
                "accelerator_telemetry_unavailable",
                required=self.required,
                reason=message,
            )
            return
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.unavailable_reason = message
            self.errors.append(message)
            mixed.trace(
                "accelerator_telemetry_unavailable",
                required=self.required,
                reason=message,
            )
            return
        assert self.paths is not None
        mixed.trace(
            "accelerator_telemetry_armed",
            active_busy_floor_percent=(
                ACCELERATOR_TELEMETRY_ACTIVE_BUSY_FLOOR_PERCENT
            ),
            advertised_max_sclk_hz=self.paths.advertised_max_sclk_hz,
            device=str(self.paths.drm_device),
            poll_interval_ms=int(HOST_GUARD_POLL_INTERVAL_SECONDS * 1000),
        )
        self.thread.start()
        self._started = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop.set()
        if self._started:
            self.thread.join(timeout=10.0)
            if self.thread.is_alive() and len(self.errors) < 8:
                self.errors.append(
                    "accelerator-telemetry thread did not stop within 10 seconds"
                )
            elif not self.errors:
                self._sample()

    def _sample(self) -> None:
        assert self.paths is not None
        try:
            sample = AcceleratorTelemetrySample(
                observed=time.monotonic(),
                busy_percent=_read_bounded_decimal(
                    self.paths.busy_percent,
                    minimum=0,
                    maximum=100,
                    label="GPU busy percent",
                ),
                sclk_hz=_read_bounded_decimal(
                    self.paths.sclk_hz,
                    minimum=0,
                    maximum=10_000_000_000,
                    label="SCLK frequency",
                ),
                power_microwatts=_read_bounded_decimal(
                    self.paths.power_microwatts,
                    minimum=0,
                    maximum=1_000_000_000,
                    label="accelerator power",
                ),
            )
            self.samples.append(sample)
        except Exception as exc:
            if len(self.errors) < 8:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self.stop.set()

    def _run(self) -> None:
        while not self.stop.wait(HOST_GUARD_POLL_INTERVAL_SECONDS):
            self._sample()
            if self.errors:
                return

    def metric_values_since(self, started: float | None) -> dict[str, float | int]:
        if not self.enabled:
            return {}
        values: dict[str, float | int] = {
            name: 0 for name in ACCELERATOR_TELEMETRY_METRIC_DEFINITIONS
        }
        values["accelerator_telemetry_error_count"] = len(self.errors)
        if self.paths is None or not self.samples:
            return values
        selected = [
            sample
            for sample in self.samples
            if started is None or sample.observed >= started
        ]
        values["accelerator_telemetry_available"] = 1
        values["accelerator_sclk_advertised_max_hz"] = (
            self.paths.advertised_max_sclk_hz
        )
        values["accelerator_telemetry_sample_count"] = len(selected)
        if not selected:
            return values
        busy = [sample.busy_percent for sample in selected]
        values["accelerator_gpu_busy_percent_p50"] = mixed.percentile_r7(busy, 0.5)
        values["accelerator_gpu_busy_percent_peak"] = max(busy)
        active = [
            sample
            for sample in selected
            if sample.busy_percent >= ACCELERATOR_TELEMETRY_ACTIVE_BUSY_FLOOR_PERCENT
        ]
        values["accelerator_telemetry_active_sample_count"] = len(active)
        if not active:
            return values
        sclk = [sample.sclk_hz for sample in active]
        power = [sample.power_microwatts for sample in active]
        values.update(
            {
                "accelerator_power_active_p50_microwatts": mixed.percentile_r7(
                    power, 0.5
                ),
                "accelerator_power_active_peak_microwatts": max(power),
                "accelerator_sclk_active_below_half_max_count": sum(
                    frequency * 2 < self.paths.advertised_max_sclk_hz
                    for frequency in sclk
                ),
                "accelerator_sclk_active_max_hz": max(sclk),
                "accelerator_sclk_active_min_hz": min(sclk),
                "accelerator_sclk_active_p50_hz": mixed.percentile_r7(sclk, 0.5),
            }
        )
        return values


DRM_MEMORY_FIELDS = (
    "drm-memory-vram",
    "drm-memory-gtt",
    "drm-memory-cpu",
)


def process_drm_memory_bytes(pid: int, proc_root: Path = Path("/proc")) -> int:
    fdinfo_dir = proc_root / str(pid) / "fdinfo"
    clients: dict[str, dict[str, int]] = {}
    saw_memory_record = False
    try:
        paths = sorted(fdinfo_dir.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise SoakError(f"cannot enumerate {fdinfo_dir}: {exc}") from exc
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            # File descriptors can close between directory enumeration and
            # read. A live DRM client remains visible through its other fd.
            continue
        client_id: str | None = None
        memory: dict[str, int] = {}
        for line in lines:
            name, separator, raw = line.partition(":")
            if not separator:
                continue
            if name == "drm-client-id":
                value = raw.strip()
                if not value:
                    raise SoakError(f"{path} has an empty drm-client-id")
                if client_id is not None and client_id != value:
                    raise SoakError(f"{path} reports multiple DRM client IDs")
                client_id = value
            elif name in DRM_MEMORY_FIELDS:
                memory[name] = parse_memory_kib(path, name, raw, "KiB")
        if not memory and client_id is None:
            continue
        if client_id is None:
            raise SoakError(f"{path} reports DRM memory without drm-client-id")
        observed = clients.setdefault(client_id, {})
        for name, value in memory.items():
            # One DRM client is commonly exposed by both render and card fds.
            # Use the largest contemporaneous value per region so duplicates
            # are not counted twice and small read-time races are conservative.
            observed[name] = max(observed.get(name, 0), value)
            saw_memory_record = True
    if not clients or not saw_memory_record:
        raise SoakError(f"server pid {pid} exposes no DRM memory accounting")
    return sum(sum(regions.values()) for regions in clients.values())


def gpu_memory_bytes(port: int, pid: int, runtime: SoakRuntime) -> int:
    if runtime.gpu_memory_scope == "server_process":
        return process_drm_memory_bytes(pid)
    if runtime.gpu_memory_scope != "device_global":
        raise SoakError(f"unsupported GPU memory scope {runtime.gpu_memory_scope!r}")
    value = mixed.parse_prometheus_used_bytes(mixed.text_request(port, "/metrics"))
    if value is None:
        raise SoakError("server metrics omitted kiln_gpu_memory_bytes{kind=\"used\"}")
    return value


class GpuMemorySampler:
    def __init__(self, port: int, pid: int, runtime: SoakRuntime) -> None:
        self.port = port
        self.pid = pid
        self.runtime = runtime
        self.stop = threading.Event()
        self.samples: list[int] = []
        self.errors: list[str] = []
        self._read_lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._run, name="qualification-gpu-memory-sampler"
        )
        self._started = False

    def start(self) -> None:
        self._sample()
        self.thread.start()
        self._started = True

    def stop_sampling(self) -> None:
        self.stop.set()
        if self._started:
            self.thread.join(timeout=10.0)
            if self.thread.is_alive() and len(self.errors) < 8:
                self.errors.append(
                    "GPU memory sampler thread did not stop within 10 seconds"
                )

    def close(self) -> None:
        self.stop_sampling()

    def read_bytes(self) -> int:
        with self._read_lock:
            if self.errors:
                raise SoakError(
                    "GPU memory sampler previously failed: " + ", ".join(self.errors)
                )
            return gpu_memory_bytes(self.port, self.pid, self.runtime)

    def _sample(self) -> None:
        try:
            self.samples.append(self.read_bytes())
        except Exception as exc:
            if len(self.errors) < 8:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self.stop.set()

    def _run(self) -> None:
        while not self.stop.wait(mixed.MEMORY_POLL_INTERVAL_SECONDS):
            self._sample()


def wait_drained(port: int, deadline: float, label: str) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        health = mixed.read_stable_health(port, deadline, label)
        runtime = health.get("decode_runtime")
        batching = runtime.get("batching_engine") if isinstance(runtime, dict) else None
        if isinstance(batching, dict):
            last = batching
            if mixed.batching_engine_drained(batching):
                return health
        time.sleep(0.05)
    raise TimeoutError(f"{label} did not drain; last batching state={last!r}")


def prefix_cache_snapshot(health: dict[str, Any]) -> dict[str, int]:
    raw = health.get("prefix_cache")
    if not isinstance(raw, dict):
        raise SoakError("health.prefix_cache is missing")
    fields = (
        "lookup_hits",
        "lookup_misses",
        "hit_tokens",
        "hit_blocks",
        "cached_blocks",
        "max_blocks",
        "cached_entries",
        "max_entries",
        "cached_state_bytes",
        "max_state_bytes",
        "active_leases",
        "pending_release_entries",
    )
    snapshot: dict[str, int] = {}
    for field in fields:
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SoakError(
                f"prefix-cache field {field} must be a nonnegative integer, got {value!r}"
            )
        snapshot[field] = value
    for used, capacity in (
        ("cached_blocks", "max_blocks"),
        ("cached_entries", "max_entries"),
        ("cached_state_bytes", "max_state_bytes"),
    ):
        if snapshot[used] > snapshot[capacity]:
            raise SoakError(
                f"prefix-cache {used}={snapshot[used]} exceeds {capacity}={snapshot[capacity]}"
            )
    return snapshot


def disabled_prefix_cache_failures(
    snapshot: dict[str, int], *, phase: str
) -> list[str]:
    failures: list[str] = []
    for field in (
        "lookup_hits",
        "lookup_misses",
        "hit_tokens",
        "hit_blocks",
        "cached_blocks",
        "max_blocks",
        "cached_entries",
        "max_entries",
        "cached_state_bytes",
        "max_state_bytes",
        "active_leases",
        "pending_release_entries",
    ):
        if snapshot[field] != 0:
            failures.append(
                f"{phase} prefix-cache {field}={snapshot[field]} while disabled"
            )
    return failures


def prefix_cache_capability_value(
    before: dict[str, float | int | bool],
    after: dict[str, float | int | bool],
) -> int:
    enabled_before = before["prefix_cache_enabled"]
    enabled_after = after["prefix_cache_enabled"]
    if not isinstance(enabled_before, bool) or not isinstance(enabled_after, bool):
        raise SoakError("prefix cache enabled capability is not boolean")
    if enabled_before != enabled_after:
        raise SoakError("prefix cache enabled capability changed during the run")
    return int(enabled_after)


VULKAN_BUFFER_FIELDS = (
    "live_device_local_buffers",
    "live_device_local_bytes",
    "live_host_visible_buffers",
    "live_host_visible_bytes",
    "peak_live_bytes",
    "device_local_allocations",
    "device_local_allocated_bytes",
    "device_local_frees",
    "device_local_freed_bytes",
    "host_visible_allocations",
    "host_visible_allocated_bytes",
    "host_visible_frees",
    "host_visible_freed_bytes",
)


def vulkan_buffer_snapshot(
    health: dict[str, Any], runtime: SoakRuntime = ROCM_RUNTIME
) -> dict[str, int] | None:
    if runtime.backend != "vulkan":
        return None
    raw = health.get("vulkan_buffers")
    if not isinstance(raw, dict):
        raise SoakError("health.vulkan_buffers is missing for the Vulkan runtime")
    if set(raw) != set(VULKAN_BUFFER_FIELDS):
        missing = sorted(set(VULKAN_BUFFER_FIELDS) - set(raw))
        extra = sorted(set(raw) - set(VULKAN_BUFFER_FIELDS))
        raise SoakError(
            f"health.vulkan_buffers field mismatch: missing={missing}, extra={extra}"
        )
    snapshot: dict[str, int] = {}
    for field in VULKAN_BUFFER_FIELDS:
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SoakError(
                f"Vulkan buffer field {field} must be a nonnegative integer, "
                f"got {value!r}"
            )
        snapshot[field] = value
    return snapshot


def vulkan_buffer_total(snapshot: dict[str, int], suffix: str) -> int:
    return snapshot[f"device_local_{suffix}"] + snapshot[f"host_visible_{suffix}"]


def vulkan_buffer_live_bytes(snapshot: dict[str, int]) -> int:
    return snapshot["live_device_local_bytes"] + snapshot["live_host_visible_bytes"]


def vulkan_buffer_live_count(snapshot: dict[str, int]) -> int:
    return snapshot["live_device_local_buffers"] + snapshot["live_host_visible_buffers"]


def vulkan_buffer_counter_delta(
    before: dict[str, int], after: dict[str, int], suffix: str
) -> int:
    start = vulkan_buffer_total(before, suffix)
    end = vulkan_buffer_total(after, suffix)
    if end < start:
        raise SoakError(f"Vulkan buffer counter {suffix} regressed: {start} -> {end}")
    return end - start


def vulkan_buffer_metric_values(
    before: dict[str, int], after: dict[str, int]
) -> dict[str, int]:
    live_start = vulkan_buffer_live_bytes(before)
    live_end = vulkan_buffer_live_bytes(after)
    return {
        "vulkan_buffer_allocated_bytes": vulkan_buffer_counter_delta(
            before, after, "allocated_bytes"
        ),
        "vulkan_buffer_allocation_count": vulkan_buffer_counter_delta(
            before, after, "allocations"
        ),
        "vulkan_buffer_free_count": vulkan_buffer_counter_delta(
            before, after, "frees"
        ),
        "vulkan_buffer_freed_bytes": vulkan_buffer_counter_delta(
            before, after, "freed_bytes"
        ),
        "vulkan_buffer_live_bytes_end": live_end,
        "vulkan_buffer_live_bytes_growth": max(0, live_end - live_start),
        "vulkan_buffer_live_bytes_start": live_start,
        "vulkan_buffer_peak_live_bytes": after["peak_live_bytes"],
    }


def vulkan_buffer_accounting_failures(
    before: dict[str, int], after: dict[str, int]
) -> list[str]:
    allocated = vulkan_buffer_counter_delta(before, after, "allocated_bytes")
    freed = vulkan_buffer_counter_delta(before, after, "freed_bytes")
    observed_bytes = vulkan_buffer_live_bytes(after) - vulkan_buffer_live_bytes(before)
    allocations = vulkan_buffer_counter_delta(before, after, "allocations")
    frees = vulkan_buffer_counter_delta(before, after, "frees")
    observed_buffers = vulkan_buffer_live_count(after) - vulkan_buffer_live_count(before)
    failures = []
    if allocated - freed != observed_bytes:
        failures.append(
            "Vulkan buffer byte accounting is inconsistent: "
            f"allocated={allocated}, freed={freed}, live_delta={observed_bytes}"
        )
    if allocations - frees != observed_buffers:
        failures.append(
            "Vulkan buffer count accounting is inconsistent: "
            f"allocations={allocations}, frees={frees}, live_delta={observed_buffers}"
        )
    return failures


VULKAN_BUFFER_POOL_NUMERIC_FIELDS = (
    "max_retained_bytes",
    "bucket_count",
    "buffer_count",
    "retained_bytes",
    "free_buffer_count",
    "free_bytes",
    "borrowed_buffer_count",
    "borrowed_bytes",
    "cache_hits",
    "cache_misses",
    "device_local_cache_misses",
    "host_visible_cache_misses",
    "eviction_count",
    "evicted_bytes",
    "uncached_allocation_count",
    "uncached_allocated_bytes",
)
VULKAN_BUFFER_POOL_FIELDS = (*VULKAN_BUFFER_POOL_NUMERIC_FIELDS, "last_cache_miss")
VULKAN_BUFFER_POOL_LAST_MISS_FIELDS = (
    "sequence",
    "route",
    "requested_bytes",
    "bucket_bytes",
    "caller_file",
    "caller_line",
)
VULKAN_BUFFER_POOL_COUNTER_FIELDS = (
    "cache_hits",
    "cache_misses",
    "device_local_cache_misses",
    "host_visible_cache_misses",
    "eviction_count",
    "evicted_bytes",
    "uncached_allocation_count",
    "uncached_allocated_bytes",
)

BATCHED_STATE_CACHE_BOOL_FIELDS = ("entry_present", "resident")
BATCHED_STATE_CACHE_GAUGE_FIELDS = (
    "capacity_rows",
    "logical_rows",
    "active_leases",
    "max_active_leases",
)
BATCHED_STATE_CACHE_COUNTER_FIELDS = (
    "take_hit_count",
    "take_miss_count",
    "take_miss_while_leased_count",
    "exact_reuse_count",
    "resident_capacity_reuse_count",
    "resident_prefix_view_count",
    "resident_refresh_count",
    "fresh_assembly_count",
    "rejected_missing_row_ids_count",
    "rejected_nonresident_rows_count",
    "rejected_nonresident_cache_count",
    "rejected_insufficient_capacity_count",
    "park_count",
    "park_replacement_eviction_count",
    "explicit_invalidation_count",
    "explicit_invalidation_eviction_count",
    "completed_row_preservation_count",
    "completed_row_eviction_count",
    "lease_drop_eviction_count",
    "resident_prefix_snapshot_suppression_count",
)
BATCHED_STATE_CACHE_FIELDS = (
    *BATCHED_STATE_CACHE_BOOL_FIELDS,
    *BATCHED_STATE_CACHE_GAUGE_FIELDS,
    *BATCHED_STATE_CACHE_COUNTER_FIELDS,
)


def batched_state_cache_snapshot(
    debug: dict[str, Any], runtime: SoakRuntime = ROCM_RUNTIME
) -> dict[str, int | bool] | None:
    if runtime.backend != "vulkan":
        return None
    caches = debug.get("caches")
    if not isinstance(caches, dict):
        raise SoakError("debug model state is missing caches")
    raw = caches.get("batched_recurrent_state")
    if not isinstance(raw, dict):
        raise SoakError("debug caches.batched_recurrent_state is missing")
    if set(raw) != set(BATCHED_STATE_CACHE_FIELDS):
        missing = sorted(set(BATCHED_STATE_CACHE_FIELDS) - set(raw))
        extra = sorted(set(raw) - set(BATCHED_STATE_CACHE_FIELDS))
        raise SoakError(
            "batched recurrent-state cache field mismatch: "
            f"missing={missing}, extra={extra}"
        )
    snapshot: dict[str, int | bool] = {}
    for field in BATCHED_STATE_CACHE_BOOL_FIELDS:
        value = raw[field]
        if not isinstance(value, bool):
            raise SoakError(
                f"batched recurrent-state cache field {field} must be boolean, "
                f"got {value!r}"
            )
        snapshot[field] = value
    for field in (*BATCHED_STATE_CACHE_GAUGE_FIELDS, *BATCHED_STATE_CACHE_COUNTER_FIELDS):
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SoakError(
                f"batched recurrent-state cache field {field} must be a nonnegative "
                f"integer, got {value!r}"
            )
        snapshot[field] = value
    if snapshot["capacity_rows"] < snapshot["logical_rows"]:
        raise SoakError("batched recurrent-state cache logical rows exceed capacity")
    if not snapshot["entry_present"] and (
        snapshot["capacity_rows"] != 0
        or snapshot["logical_rows"] != 0
        or snapshot["resident"]
    ):
        raise SoakError("absent batched recurrent-state cache carries parked ownership")
    if snapshot["max_active_leases"] < snapshot["active_leases"]:
        raise SoakError("batched recurrent-state active leases exceed the process peak")
    return snapshot


RESIDENT_RECURRENT_STATE_FIELDS = (
    "entry_count",
    "buffer_bytes",
    "allocation_bytes",
)


def resident_recurrent_state_snapshot(
    debug: dict[str, Any], runtime: SoakRuntime = ROCM_RUNTIME
) -> dict[str, int] | None:
    if runtime.backend != "vulkan":
        return None
    caches = debug.get("caches")
    if not isinstance(caches, dict):
        raise SoakError("debug model state is missing caches")
    raw = caches.get("resident_recurrent_state")
    if not isinstance(raw, dict):
        raise SoakError("debug caches.resident_recurrent_state is missing")
    if set(raw) != set(RESIDENT_RECURRENT_STATE_FIELDS):
        missing = sorted(set(RESIDENT_RECURRENT_STATE_FIELDS) - set(raw))
        extra = sorted(set(raw) - set(RESIDENT_RECURRENT_STATE_FIELDS))
        raise SoakError(
            "resident recurrent-state field mismatch: "
            f"missing={missing}, extra={extra}"
        )
    snapshot: dict[str, int] = {}
    for field in RESIDENT_RECURRENT_STATE_FIELDS:
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SoakError(
                f"resident recurrent-state field {field} must be a nonnegative "
                f"integer, got {value!r}"
            )
        snapshot[field] = value
    if snapshot["allocation_bytes"] < snapshot["buffer_bytes"]:
        raise SoakError("resident recurrent-state allocation bytes are below buffer bytes")
    if snapshot["entry_count"] == 0 and (
        snapshot["buffer_bytes"] != 0 or snapshot["allocation_bytes"] != 0
    ):
        raise SoakError("empty resident recurrent-state registry retains bytes")
    if snapshot["entry_count"] > 0 and snapshot["buffer_bytes"] == 0:
        raise SoakError("nonempty resident recurrent-state registry reports zero bytes")
    return snapshot


def resident_recurrent_state_drain_failures(
    snapshot: dict[str, int] | None, phase: str
) -> list[str]:
    if snapshot is None:
        return []
    return [
        f"{phase} retained backend-private recurrent state after drain: "
        f"entries={snapshot['entry_count']}, buffer_bytes={snapshot['buffer_bytes']}, "
        f"allocation_bytes={snapshot['allocation_bytes']}"
    ] if any(snapshot.values()) else []


def resident_recurrent_state_metric_values(
    snapshot: dict[str, int],
) -> dict[str, int]:
    return {
        "resident_recurrent_state_entries_end": snapshot["entry_count"],
        "resident_recurrent_state_buffer_bytes_end": snapshot["buffer_bytes"],
        "resident_recurrent_state_allocation_bytes_end": snapshot["allocation_bytes"],
    }


def batched_state_cache_counter_delta(
    before: dict[str, int | bool], after: dict[str, int | bool], field: str
) -> int:
    start = before[field]
    end = after[field]
    if isinstance(start, bool) or isinstance(end, bool):
        raise SoakError(f"batched recurrent-state cache counter {field} is boolean")
    if end < start:
        raise SoakError(
            f"batched recurrent-state cache counter {field} regressed: {start} -> {end}"
        )
    return end - start


def batched_state_cache_trace_values(
    before: dict[str, int | bool], after: dict[str, int | bool]
) -> dict[str, int | bool]:
    values: dict[str, int | bool] = {
        "batched_state_cache_entry_present": after["entry_present"],
        "batched_state_cache_capacity_rows": after["capacity_rows"],
        "batched_state_cache_logical_rows": after["logical_rows"],
        "batched_state_cache_resident": after["resident"],
        "batched_state_cache_active_leases": after["active_leases"],
        "batched_state_cache_max_active_leases": after["max_active_leases"],
    }
    for field in BATCHED_STATE_CACHE_COUNTER_FIELDS:
        values[f"batched_state_cache_{field}_delta"] = (
            batched_state_cache_counter_delta(before, after, field)
        )
    return values


def batched_state_cache_metric_values(
    before: dict[str, int | bool], after: dict[str, int | bool]
) -> dict[str, int]:
    values = {
        "batched_state_cache_active_leases_end": int(after["active_leases"]),
        "batched_state_cache_capacity_rows_end": int(after["capacity_rows"]),
        "batched_state_cache_entry_present_end": int(after["entry_present"]),
        "batched_state_cache_logical_rows_end": int(after["logical_rows"]),
        "batched_state_cache_max_active_leases": int(after["max_active_leases"]),
        "batched_state_cache_resident_end": int(after["resident"]),
    }
    for field in BATCHED_STATE_CACHE_COUNTER_FIELDS:
        values[f"batched_state_cache_{field}"] = batched_state_cache_counter_delta(
            before, after, field
        )
    return values


def vulkan_buffer_pool_snapshot(
    health: dict[str, Any], runtime: SoakRuntime = ROCM_RUNTIME
) -> dict[str, Any] | None:
    if runtime.backend != "vulkan":
        return None
    raw = health.get("vulkan_buffer_pool")
    if not isinstance(raw, dict):
        raise SoakError("health.vulkan_buffer_pool is missing for the Vulkan runtime")
    if set(raw) != set(VULKAN_BUFFER_POOL_FIELDS):
        missing = sorted(set(VULKAN_BUFFER_POOL_FIELDS) - set(raw))
        extra = sorted(set(raw) - set(VULKAN_BUFFER_POOL_FIELDS))
        raise SoakError(
            "health.vulkan_buffer_pool field mismatch: "
            f"missing={missing}, extra={extra}"
        )
    snapshot: dict[str, Any] = {}
    for field in VULKAN_BUFFER_POOL_NUMERIC_FIELDS:
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SoakError(
                f"Vulkan buffer-pool field {field} must be a nonnegative integer, "
                f"got {value!r}"
            )
        snapshot[field] = value
    last_miss = raw["last_cache_miss"]
    if last_miss is None:
        if snapshot["cache_misses"] != 0:
            raise SoakError("Vulkan buffer pool omitted its last cache miss")
        snapshot["last_cache_miss"] = None
    else:
        if not isinstance(last_miss, dict):
            raise SoakError("Vulkan buffer-pool last_cache_miss must be an object")
        if set(last_miss) != set(VULKAN_BUFFER_POOL_LAST_MISS_FIELDS):
            missing = sorted(set(VULKAN_BUFFER_POOL_LAST_MISS_FIELDS) - set(last_miss))
            extra = sorted(set(last_miss) - set(VULKAN_BUFFER_POOL_LAST_MISS_FIELDS))
            raise SoakError(
                "Vulkan buffer-pool last_cache_miss field mismatch: "
                f"missing={missing}, extra={extra}"
            )
        for field in ("sequence", "requested_bytes", "bucket_bytes", "caller_line"):
            value = last_miss[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SoakError(
                    f"Vulkan buffer-pool last_cache_miss.{field} must be a positive "
                    f"integer, got {value!r}"
                )
        if last_miss["route"] not in {"device_local", "host_visible"}:
            raise SoakError(
                "Vulkan buffer-pool last_cache_miss.route must be device_local or "
                f"host_visible, got {last_miss['route']!r}"
            )
        if not isinstance(last_miss["caller_file"], str) or not last_miss["caller_file"]:
            raise SoakError(
                "Vulkan buffer-pool last_cache_miss.caller_file must be non-empty"
            )
        if last_miss["sequence"] != snapshot["cache_misses"]:
            raise SoakError(
                "Vulkan buffer-pool last cache-miss sequence does not match its counter"
            )
        if last_miss["requested_bytes"] > last_miss["bucket_bytes"]:
            raise SoakError("Vulkan buffer-pool last cache-miss bucket rounded down")
        snapshot["last_cache_miss"] = dict(last_miss)
    if (
        snapshot["device_local_cache_misses"]
        + snapshot["host_visible_cache_misses"]
        != snapshot["cache_misses"]
    ):
        raise SoakError("Vulkan buffer-pool route miss counters do not reconcile")
    if snapshot["retained_bytes"] > snapshot["max_retained_bytes"]:
        raise SoakError(
            "Vulkan buffer pool exceeds its configured cap: "
            f"{snapshot['retained_bytes']} > {snapshot['max_retained_bytes']} bytes"
        )
    expected_max_retained_bytes = round(runtime.vulkan_buffer_pool_gb * 1024**3)
    if snapshot["max_retained_bytes"] != expected_max_retained_bytes:
        raise SoakError(
            "Vulkan buffer-pool configured cap mismatch: "
            f"{snapshot['max_retained_bytes']} != {expected_max_retained_bytes} bytes"
        )
    if snapshot["free_bytes"] + snapshot["borrowed_bytes"] != snapshot["retained_bytes"]:
        raise SoakError("Vulkan buffer-pool byte ownership does not reconcile")
    if (
        snapshot["free_buffer_count"] + snapshot["borrowed_buffer_count"]
        != snapshot["buffer_count"]
    ):
        raise SoakError("Vulkan buffer-pool count ownership does not reconcile")
    return snapshot


def vulkan_buffer_pool_counter_delta(
    before: dict[str, Any], after: dict[str, Any], field: str
) -> int:
    start = int(before[field])
    end = int(after[field])
    if end < start:
        raise SoakError(f"Vulkan buffer-pool counter {field} regressed: {start} -> {end}")
    return end - start


def stabilization_cycle_is_stable(
    runtime: SoakRuntime,
    *,
    gpu_delta: int,
    rss_delta: int,
    vulkan_live_bytes_delta: int = 0,
    vulkan_allocation_count: int = 0,
    vulkan_free_count: int = 0,
    vulkan_pool_cache_miss_count: int = 0,
    vulkan_pool_eviction_count: int = 0,
    vulkan_pool_uncached_allocation_count: int = 0,
) -> bool:
    if gpu_delta > runtime.stabilization_gpu_delta_limit_bytes:
        return False
    if runtime.backend != "vulkan":
        return rss_delta <= runtime.stabilization_rss_delta_limit_bytes
    return all(
        value == 0
        for value in (
            vulkan_live_bytes_delta,
            vulkan_allocation_count,
            vulkan_free_count,
            vulkan_pool_cache_miss_count,
            vulkan_pool_eviction_count,
            vulkan_pool_uncached_allocation_count,
        )
    )


def stabilization_gpu_growth_delta(
    runtime: SoakRuntime,
    *,
    current_gpu: int,
    previous_gpu: int,
    stabilization_gpu_high_water: int,
) -> int:
    comparison = (
        stabilization_gpu_high_water
        if runtime.backend == "cuda"
        else previous_gpu
    )
    return max(0, current_gpu - comparison)


def measurement_gpu_baseline(
    runtime: SoakRuntime,
    *,
    current_gpu: int,
    stabilization_gpu_high_water: int,
) -> int:
    if runtime.backend == "cuda":
        return max(current_gpu, stabilization_gpu_high_water)
    return current_gpu


def vulkan_buffer_pool_miss_trace_values(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    total = vulkan_buffer_pool_counter_delta(before, after, "cache_misses")
    device_local = vulkan_buffer_pool_counter_delta(
        before, after, "device_local_cache_misses"
    )
    host_visible = vulkan_buffer_pool_counter_delta(
        before, after, "host_visible_cache_misses"
    )
    if device_local + host_visible != total:
        raise SoakError("Vulkan buffer-pool interval route misses do not reconcile")
    values: dict[str, Any] = {
        "vulkan_pool_cache_miss_count": total,
        "vulkan_pool_device_local_cache_miss_count": device_local,
        "vulkan_pool_host_visible_cache_miss_count": host_visible,
    }
    if total != 0:
        last_miss = after["last_cache_miss"]
        if not isinstance(last_miss, dict):
            raise SoakError("Vulkan buffer-pool miss interval lacks last-miss attribution")
        values["vulkan_pool_last_cache_miss"] = dict(last_miss)
    return values


def vulkan_buffer_pool_metric_values(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, int]:
    if before["max_retained_bytes"] != after["max_retained_bytes"]:
        raise SoakError("Vulkan buffer-pool limit changed after startup")
    return {
        "vulkan_buffer_pool_cache_hit_count": vulkan_buffer_pool_counter_delta(
            before, after, "cache_hits"
        ),
        "vulkan_buffer_pool_cache_miss_count": vulkan_buffer_pool_counter_delta(
            before, after, "cache_misses"
        ),
        "vulkan_buffer_pool_device_local_cache_miss_count": (
            vulkan_buffer_pool_counter_delta(
                before, after, "device_local_cache_misses"
            )
        ),
        "vulkan_buffer_pool_evicted_bytes": vulkan_buffer_pool_counter_delta(
            before, after, "evicted_bytes"
        ),
        "vulkan_buffer_pool_eviction_count": vulkan_buffer_pool_counter_delta(
            before, after, "eviction_count"
        ),
        "vulkan_buffer_pool_host_visible_cache_miss_count": (
            vulkan_buffer_pool_counter_delta(
                before, after, "host_visible_cache_misses"
            )
        ),
        "vulkan_buffer_pool_limit_bytes": int(after["max_retained_bytes"]),
        "vulkan_buffer_pool_retained_bytes_end": int(after["retained_bytes"]),
        "vulkan_buffer_pool_retained_bytes_growth": max(
            0, int(after["retained_bytes"]) - int(before["retained_bytes"])
        ),
        "vulkan_buffer_pool_retained_bytes_start": int(before["retained_bytes"]),
        "vulkan_buffer_pool_uncached_allocated_bytes": vulkan_buffer_pool_counter_delta(
            before, after, "uncached_allocated_bytes"
        ),
        "vulkan_buffer_pool_uncached_allocation_count": vulkan_buffer_pool_counter_delta(
            before, after, "uncached_allocation_count"
        ),
    }


def unaccounted_blocks(
    batching: dict[str, float | int], prefix_cache: dict[str, int]
) -> int:
    used = batching["blocks_used"]
    cached = prefix_cache["cached_blocks"]
    if not isinstance(used, int):
        raise SoakError(f"scheduler blocks_used is not an integer: {used!r}")
    if cached > used:
        raise SoakError(
            f"prefix cache accounts for {cached} blocks but scheduler reports {used} used"
        )
    return used - cached


def graph_warmup_ready(
    graph: dict[str, int], runtime: SoakRuntime = ROCM_RUNTIME
) -> bool:
    if runtime.graph_execution_required:
        return (
            graph["capture_successes"] >= 1
            and graph["replay_successes"] >= 1
            and graph["failures"] == 0
        )
    return all(value == 0 for value in graph.values())


def run_wave(
    port: int,
    *,
    wave: int,
    base_seed: int,
    deadline: float,
    phase: str = "measured",
    prompt_epoch: int | None = None,
    runtime: SoakRuntime = ROCM_RUNTIME,
    worker_evidence: RequestWorkerEvidence | None = None,
) -> list[mixed.StreamResult]:
    concurrency = runtime.wave_concurrency[wave % len(runtime.wave_concurrency)]
    abort = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    futures: list[concurrent.futures.Future[mixed.StreamResult]] = []
    results: list[mixed.StreamResult] | None = None
    primary_error: Exception | None = None
    unfinished: set[concurrent.futures.Future[mixed.StreamResult]] = set()
    try:
        for slot in range(concurrency):
            role = f"soak-{phase}-w{wave:05d}-r{slot:02d}"
            prompt_role = (
                f"soak-shared-r{slot:02d}"
                if prompt_epoch is None
                else f"soak-warm-e{prompt_epoch:05d}-r{slot:02d}"
            )
            futures.append(
                pool.submit(
                    mixed.run_stream,
                    port,
                    name=role,
                    marker=mixed.workload_marker(base_seed, prompt_role),
                    prompt_words=prompt_words_for_wave(runtime, wave, slot),
                    max_tokens=runtime.max_tokens,
                    seed=base_seed + wave * 100 + slot,
                    absolute_deadline=deadline,
                    abort_event=abort,
                    request_timeout_seconds=runtime.request_timeout_seconds,
                )
            )
        results = [
            future.result(
                timeout=mixed.remaining_until(deadline, f"soak {phase} wave")
            )
            for future in futures
        ]
    except Exception as exc:
        primary_error = exc
    finally:
        abort.set()
        for future in futures:
            future.cancel()
        _, unfinished = concurrent.futures.wait(
            futures,
            timeout=REQUEST_WORKER_CLEANUP_TIMEOUT_SECONDS,
        )
        if worker_evidence is not None:
            worker_evidence.observe(len(unfinished))
        pool.shutdown(wait=False, cancel_futures=True)
    if primary_error is not None:
        if unfinished:
            raise SoakError(
                f"{type(primary_error).__name__}: {primary_error}; "
                f"{len(unfinished)} request workers survived wave cleanup"
            ) from primary_error
        raise primary_error
    if unfinished:
        raise SoakError(f"{len(unfinished)} request workers survived wave cleanup")
    assert results is not None
    return results


def record_drained_warmup_wave(
    request_count: int,
    wave_count: int,
    results: list[mixed.StreamResult],
) -> tuple[int, int]:
    return request_count + len(results), wave_count + 1


def prompt_words_for_wave(runtime: SoakRuntime, wave: int, slot: int) -> int:
    if runtime.prompt_assignment == SLOT_BY_REQUEST:
        return runtime.prompt_words[slot]
    if runtime.prompt_assignment == COHORT_BY_CYCLE:
        cycle = wave // len(runtime.wave_concurrency)
        return runtime.prompt_words[cycle % len(runtime.prompt_words)]
    raise SoakError(f"unsupported prompt assignment: {runtime.prompt_assignment!r}")


def invalid_stream_result_summary(
    result: mixed.StreamResult, expected_completion_tokens: int
) -> str:
    violations = stream_result_violations(result, expected_completion_tokens)
    oracle_failure = mixed.deterministic_response_oracle_failure(result)
    return (
        f"{result.name}(violations={'+'.join(violations) or 'none'},"
        f"error={result.error!r},finish_reason={result.finish_reason!r},"
        f"prompt_tokens={result.prompt_tokens},"
        f"completion_tokens={result.completion_tokens},"
        f"usage_records={result.usage_records},"
        f"token_timings={len(result.token_ready_times)},"
        f"token_ids={result.token_ids!r},"
        f"resident_prefill_used={result.resident_prefill_used!r},"
        f"semantic_events={len(result.semantic_times)},done={result.done},"
        f"cancelled={result.cancelled},response_oracle_failure={oracle_failure!r},"
        f"response_text={mixed.bounded_response_text(result)},"
        f"actor_queue_ms={result.actor_queue_ms!r},"
        f"actor_admission_ms={result.actor_admission_ms!r},"
        f"actor_prefill_wall_ms={result.actor_prefill_wall_ms!r})"
    )


def stream_result_violations(
    result: mixed.StreamResult, expected_completion_tokens: int
) -> list[str]:
    violations: list[str] = []
    if not result.success:
        violations.append("success=false")
    if result.finish_reason != "length":
        violations.append("finish_reason!=length")
    if result.completion_tokens != expected_completion_tokens:
        violations.append(f"completion_tokens!={expected_completion_tokens}")
    if mixed.deterministic_response_oracle_failure(result) is not None:
        violations.append("response_oracle_failed")
    return violations


def valid_stream_result(
    result: mixed.StreamResult, expected_completion_tokens: int
) -> bool:
    return not stream_result_violations(result, expected_completion_tokens)


@dataclasses.dataclass(frozen=True)
class MeasurementResultEvidence:
    values: dict[str, float | int]
    successes: list[mixed.StreamResult]
    attributed_itl_outliers: int
    unexplained_itl_outliers: int


def measurement_result_evidence(
    *,
    warmup_itl_ms: list[float],
    results: list[mixed.StreamResult],
    measurement_events: list[mixed.ObservedEvent],
    all_server_events: list[mixed.ObservedEvent],
    expected_completion_tokens: int,
    cancellation_count: int,
    duration_seconds: float,
    wave_count: int,
    steady_state_warmup_request_count: int,
    steady_state_warmup_wave_count: int,
) -> MeasurementResultEvidence:
    successes = [
        result
        for result in results
        if valid_stream_result(result, expected_completion_tokens)
    ]
    outliers = mixed.classify_itl_outliers(
        warmup_itl_ms, successes, measurement_events
    )
    itls = [gap for result in successes for gap in result.itl_ms]
    ttfts = [result.ttft_ms for result in successes]
    prompt_tokens = [result.prompt_tokens for result in successes]
    completion_tokens = sum(result.completion_tokens for result in successes)
    values: dict[str, float | int] = {
        "attributed_itl_outlier_count": outliers.attributed,
        "cancellation_confirmed_count": cancellation_count,
        "completion_token_count": completion_tokens,
        "device_fault_event_count": sum(
            event.category == "device_fault" for event in all_server_events
        ),
        "itl_ms_p50": mixed.percentile_r7(itls, 0.5),
        "itl_ms_p99": mixed.percentile_r7(itls, 0.99),
        "itl_ms_p999": mixed.percentile_r7(itls, 0.999),
        "non_finite_response_count": sum(
            result.error is not None and "non-finite" in result.error.lower()
            for result in results
        ),
        "output_token_throughput_per_second": (
            completion_tokens / duration_seconds if duration_seconds > 0 else 0.0
        ),
        "prompt_tokens_max": max(prompt_tokens, default=0),
        "prompt_tokens_min": min(prompt_tokens, default=0),
        "request_failure_count": len(results) - len(successes),
        "request_count": len(results),
        "soak_duration_seconds": duration_seconds,
        "steady_state_warmup_request_count": steady_state_warmup_request_count,
        "steady_state_warmup_wave_count": steady_state_warmup_wave_count,
        "ttft_ms_p50": mixed.percentile_r7(ttfts, 0.5),
        "ttft_ms_p99": mixed.percentile_r7(ttfts, 0.99),
        "ttft_ms_p999": mixed.percentile_r7(ttfts, 0.999),
        "unexplained_itl_outlier_count": outliers.unexplained,
        "wave_count": wave_count,
        "zero_token_response_count": sum(
            result.completion_tokens == 0 for result in results
        ),
    }
    values.update(mixed.latency_phase_metric_values(successes))
    return MeasurementResultEvidence(
        values,
        successes,
        outliers.attributed,
        outliers.unexplained,
    )


def invalid_stream_results_summary(
    results: list[mixed.StreamResult], expected_completion_tokens: int
) -> str:
    return ", ".join(
        invalid_stream_result_summary(item, expected_completion_tokens)
        for item in results[:8]
    )


def run_cancellation(
    port: int,
    *,
    wave: int,
    base_seed: int,
    phase: str,
    deadline: float,
    runtime: SoakRuntime = ROCM_RUNTIME,
) -> str | None:
    role = f"soak-{phase}-cancel-w{wave:05d}"
    wave_seed = base_seed + wave * 100
    cancelled = mixed.run_stream(
        port,
        name=role,
        marker=mixed.workload_marker(wave_seed, role),
        prompt_words=runtime.cancellation_prompt_words,
        max_tokens=runtime.cancellation_max_tokens,
        seed=wave_seed + 99,
        cancel_after=mixed.CANCELLATION_AFTER_DELTAS,
        absolute_deadline=deadline,
        request_timeout_seconds=runtime.request_timeout_seconds,
    )
    confirmed, _ = mixed.wait_for_cancellation_and_drain(
        port, cancelled.marker, deadline
    )
    failures: list[str] = []
    if (
        not cancelled.cancelled
        or len(cancelled.semantic_times) < mixed.CANCELLATION_AFTER_DELTAS
        or not confirmed
    ):
        failures.append(f"cancellation was not confirmed in {phase} wave {wave}")
    oracle_failure = mixed.deterministic_response_oracle_failure(cancelled)
    if oracle_failure is not None:
        failures.append(
            f"cancellation failed {mixed.RESPONSE_ORACLE} in {phase} wave {wave}: "
            f"{oracle_failure}"
        )
    if not failures:
        return None
    return (
        "; ".join(failures)
        + f"; response_text={mixed.bounded_response_text(cancelled)}"
    )


def metric_definitions(
    runtime: SoakRuntime = ROCM_RUNTIME,
) -> dict[str, tuple[str, str, bool]]:
    if runtime.backend == "rocm":
        return ROCM_METRIC_DEFINITIONS
    if runtime.backend == "vulkan":
        return VULKAN_METRIC_DEFINITIONS
    if runtime.backend == "cuda":
        return CUDA_METRIC_DEFINITIONS
    raise SoakError(f"unsupported soak metric backend {runtime.backend!r}")


def resident_prefill_metric_values(
    before: dict[str, int | bool], after: dict[str, int | bool]
) -> dict[str, int]:
    enabled_before = before["resident_prefill_enabled"]
    enabled_after = after["resident_prefill_enabled"]
    if not isinstance(enabled_before, bool) or not isinstance(enabled_after, bool):
        raise SoakError("resident prefill enabled capability is not boolean")
    if enabled_before != enabled_after:
        raise SoakError("resident prefill enabled capability changed during the run")
    max_batch_size = after["max_resident_prefill_batch_size"]
    if max_batch_size < before["max_resident_prefill_batch_size"]:
        raise SoakError("resident prefill maximum batch size regressed")
    return {
        "resident_prefill_enabled": int(enabled_after),
        "resident_prefill_active_rows_end": after["active_resident_prefill"],
        "resident_prefill_attempt_count": mixed.counter_delta(
            before, after, "total_resident_prefill_attempts"
        ),
        "resident_prefill_completed_row_count": mixed.counter_delta(
            before, after, "total_resident_prefill_completed_rows"
        ),
        "resident_prefill_forward_count": mixed.counter_delta(
            before, after, "total_resident_prefill_forwards"
        ),
        "resident_prefill_initial_decline_count": mixed.counter_delta(
            before, after, "total_resident_prefill_initial_declines"
        ),
        "resident_prefill_max_batch_size": max_batch_size,
        "resident_prefill_route_failure_count": mixed.counter_delta(
            before, after, "total_resident_prefill_route_failures"
        ),
        "resident_prefill_row_count": mixed.counter_delta(
            before, after, "total_resident_prefill_rows"
        ),
    }


def stabilization_resident_prefill_metric_values(
    before: dict[str, int | bool], after: dict[str, int | bool]
) -> dict[str, int]:
    return {
        f"stabilization_{name}": value
        for name, value in resident_prefill_metric_values(before, after).items()
    }


def partial_stabilization_resident_prefill_metric_values(
    runtime: SoakRuntime,
    before: dict[str, int | bool] | None,
    after: dict[str, int | bool] | None,
) -> dict[str, int]:
    if runtime.backend != "vulkan" or before is None or after is None:
        return {}
    return stabilization_resident_prefill_metric_values(before, after)


def resident_prefill_contract_failures(
    values: dict[str, int], *, max_configured_rows: int
) -> list[str]:
    failures: list[str] = []
    for name in (
        "resident_prefill_active_rows_end",
        "resident_prefill_initial_decline_count",
        "resident_prefill_route_failure_count",
    ):
        if values[name] != 0:
            failures.append(f"{name}={values[name]}, expected 0")

    attempts = values["resident_prefill_attempt_count"]
    forwards = values["resident_prefill_forward_count"]
    declines = values["resident_prefill_initial_decline_count"]
    route_failures = values["resident_prefill_route_failure_count"]
    rows = values["resident_prefill_row_count"]
    completed_rows = values["resident_prefill_completed_row_count"]
    max_batch_size = values["resident_prefill_max_batch_size"]
    if values["resident_prefill_enabled"] == 0:
        for name in (
            "resident_prefill_attempt_count",
            "resident_prefill_completed_row_count",
            "resident_prefill_forward_count",
            "resident_prefill_initial_decline_count",
            "resident_prefill_max_batch_size",
            "resident_prefill_route_failure_count",
            "resident_prefill_row_count",
        ):
            if values[name] != 0:
                failures.append(f"{name}={values[name]} while resident prefill is disabled")
        return failures
    if route_failures == 0 and attempts != forwards + declines:
        failures.append(
            "resident prefill attempts do not reconcile with forwards and declines: "
            f"{attempts} != {forwards} + {declines}"
        )
    if forwards < 1:
        failures.append("soak completed without a measured resident prefill forward")
    if rows <= forwards:
        failures.append(
            "resident prefill row count did not prove a measured multi-row forward: "
            f"{rows} <= {forwards}"
        )
    if completed_rows < 1 or completed_rows > rows:
        failures.append(
            "resident prefill completed rows were outside the measured row range: "
            f"completed={completed_rows}, rows={rows}"
        )
    if max_batch_size < 2:
        failures.append(
            "soak completed without a measured multi-row resident prefill batch"
        )
    if max_batch_size > max_configured_rows:
        failures.append(
            "resident prefill maximum batch size exceeded configured concurrency: "
            f"{max_batch_size} > {max_configured_rows}"
        )
    return failures


def metrics_from_values(
    values: dict[str, float | int], runtime: SoakRuntime = ROCM_RUNTIME
) -> list[dict[str, Any]]:
    definitions = metric_definitions(runtime)
    if set(values) != set(definitions):
        missing = sorted(set(definitions) - set(values))
        extra = sorted(set(values) - set(definitions))
        raise SoakError(f"metric set mismatch: missing={missing}, extra={extra}")
    metrics: list[dict[str, Any]] = []
    for name in sorted(values):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SoakError(f"metric {name} is not numeric: {value!r}")
        if not math.isfinite(value) or value < 0:
            raise SoakError(f"metric {name} is not finite and nonnegative: {value!r}")
        unit, aggregation, lower_is_better = definitions[name]
        metrics.append(
            {
                "name": name,
                "value": value,
                "unit": unit,
                "aggregation": aggregation,
                "lower_is_better": lower_is_better,
            }
        )
    return metrics


def execute(
    model_path: Path,
    seed: int,
    minimum_duration_seconds: float,
    memory_growth_limit_bytes: int,
    runtime: SoakRuntime = ROCM_RUNTIME,
) -> tuple[list[dict[str, Any]], str | None]:
    binary, binary_hash, build_seconds, runtime_started, setup_deadline = (
        build_binary_for_soak(runtime)
    )
    mixed.trace(
        "soak_binary_built",
        build_seconds=build_seconds,
        path=str(binary.relative_to(ROOT)),
        sha256=binary_hash,
    )
    port = mixed.free_loopback_port()
    run_dir = mixed.create_serving_run_dir(f"{runtime.backend}-soak")
    adapter_dir = run_dir / "adapters"
    snapshot_dir = run_dir / "model-snapshots"
    config_path = run_dir / "kiln.toml"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    mixed.write_server_config(
        config_path,
        runtime.variant_id,
        model_path,
        port,
        adapter_dir,
        snapshot_dir,
        rocm_graph_cache_entries=runtime.graph_cache_max,
        inference_memory_fraction=runtime.inference_memory_fraction,
        memory_floor_gb=runtime.memory_floor_gb,
    )
    process, server_log = mixed.start_server(
        binary, config_path, runtime.variant_id, runtime.build_spec
    )
    try:
        sampler = GpuMemorySampler(port, process.pid, runtime)
    except Exception:
        try:
            mixed.terminate_process(process)
        finally:
            server_log.join()
            shutil.rmtree(run_dir, ignore_errors=True)
        raise
    accelerator_sampler = AcceleratorTelemetrySampler(
        enabled=runtime.accelerator_telemetry_enabled,
        required=runtime.accelerator_telemetry_required,
    )
    host_guard = (
        HostMemoryGuard(process, runtime.host_mem_available_floor_bytes)
        if runtime.host_mem_available_floor_bytes is not None
        else None
    )
    if host_guard is not None:
        host_guard.start()
    accelerator_sampler.start()
    shutdown: mixed.ShutdownOutcome | None = None
    snapshot_residue: list[str] = []
    values: dict[str, float | int] | None = None
    measurement_started: float | None = None
    failures: list[str] = []
    stabilization_requests = 0
    stabilization_cancellations = 0
    stabilization_cycles = 0
    stabilization_stable_cycles = 0
    stabilization_final_gpu_delta = 0
    stabilization_final_rss_delta = 0
    stabilization_max_gpu_delta = 0
    stabilization_max_rss_delta = 0
    stabilization_rss_growth = 0
    stabilization_vulkan_growth_cycles = 0
    worker_evidence = RequestWorkerEvidence()
    observed_stabilization_batching_start: dict[str, int | bool] | None = None
    observed_stabilization_batching_end: dict[str, int | bool] | None = None
    observed_vulkan_buffers_start: dict[str, int] | None = None
    observed_vulkan_buffers_end: dict[str, int] | None = None
    observed_vulkan_pool_start: dict[str, Any] | None = None
    observed_vulkan_pool_end: dict[str, Any] | None = None
    observed_batched_state_start: dict[str, int | bool] | None = None
    observed_batched_state_end: dict[str, int | bool] | None = None
    observed_process_mappings_start: ProcessMemoryMappingSnapshot | None = None
    observed_process_mappings_end: ProcessMemoryMappingSnapshot | None = None
    warmup: mixed.StreamResult | None = None
    health_start: dict[str, Any] | None = None
    health_end: dict[str, Any] | None = None
    graph_start: dict[str, int] | None = None
    graph_end: dict[str, int] | None = None
    batching_start: dict[str, int | bool] | None = None
    batching_end: dict[str, int | bool] | None = None
    prefix_start: dict[str, int] | None = None
    prefix_end: dict[str, int] | None = None
    vulkan_buffers_start: dict[str, int] | None = None
    vulkan_buffers_end: dict[str, int] | None = None
    vulkan_pool_start: dict[str, Any] | None = None
    vulkan_pool_end: dict[str, Any] | None = None
    batched_state_start: dict[str, int | bool] | None = None
    batched_state_end: dict[str, int | bool] | None = None
    resident_state_end: dict[str, int] | None = None
    gpu_start: int | None = None
    gpu_end: int | None = None
    rss_start: int | None = None
    rss_end: int | None = None
    all_results: list[mixed.StreamResult] = []
    rss_samples: list[int] = []
    wave = 0
    cancellations = 0
    steady_state_warmup_requests = 0
    steady_state_warmup_waves = 0
    try:
        mixed.wait_ready(
            port,
            process,
            server_log,
            setup_deadline,
            require_prewarm_log_evidence=runtime.backend != "cuda",
        )
        health_startup = mixed.read_stable_health(
            port, setup_deadline, "soak startup health"
        )
        debug_start = mixed.json_request(port, "GET", "/v1/debug/model-state")
        batched_state_cache_snapshot(debug_start, runtime)
        resident_startup = resident_recurrent_state_snapshot(debug_start, runtime)
        failures.extend(
            resident_recurrent_state_drain_failures(resident_startup, "startup")
        )
        failures.extend(
            mixed.attest_runtime(
                runtime.variant_id,
                health_startup,
                debug_start,
                rocm_graph_cache_entries=runtime.graph_cache_max,
            )
        )
        for attempt in range(mixed.MAX_WARMUP_REQUESTS):
            warmup = mixed.run_stream(
                port,
                name=f"soak-warmup-{attempt + 1}",
                marker=mixed.workload_marker(seed, f"soak-warmup-{attempt + 1}"),
                prompt_words=16 + attempt * 8,
                max_tokens=mixed.WARMUP_MAX_TOKENS,
                seed=seed + attempt,
                absolute_deadline=setup_deadline,
                request_timeout_seconds=runtime.request_timeout_seconds,
            )
            if not valid_stream_result(warmup, mixed.WARMUP_MAX_TOKENS):
                raise SoakError(
                    "warmup failed: "
                    + invalid_stream_result_summary(
                        warmup, mixed.WARMUP_MAX_TOKENS
                    )
                )
            health_start = wait_drained(port, setup_deadline, "soak warmup")
            graph = mixed.graph_snapshot(health_start)
            if graph_warmup_ready(graph, runtime):
                break
        else:
            requirement = (
                "capture and replay" if runtime.graph_execution_required else "remain disabled"
            )
            raise SoakError(f"graph warmup did not {requirement}")
        assert warmup is not None and health_start is not None

        batching_warm = mixed.batching_snapshot(health_start)
        prefix_cache_enabled = batching_warm["prefix_cache_enabled"]
        assert isinstance(prefix_cache_enabled, bool)
        prefix_warm = prefix_cache_snapshot(health_start)
        if not prefix_cache_enabled:
            disabled_failures = disabled_prefix_cache_failures(
                prefix_warm, phase="initial warmup"
            )
            if disabled_failures:
                raise SoakError("; ".join(disabled_failures))
        while (
            prefix_cache_enabled
            and prefix_warm["cached_entries"] < prefix_warm["max_entries"]
        ):
            if steady_state_warmup_waves >= runtime.max_steady_state_warmup_waves:
                raise SoakError(
                    "prefix cache did not reach steady-state capacity within "
                    f"{runtime.max_steady_state_warmup_waves} warmup waves"
                )
            warm_results = run_wave(
                port,
                wave=steady_state_warmup_waves,
                base_seed=seed + 1_000_000,
                deadline=setup_deadline,
                phase="warmup",
                prompt_epoch=steady_state_warmup_waves,
                runtime=runtime,
                worker_evidence=worker_evidence,
            )
            bad_warm = [
                result
                for result in warm_results
                if not valid_stream_result(result, runtime.max_tokens)
            ]
            if bad_warm:
                raise SoakError(
                    "steady-state warmup produced invalid responses: "
                    + invalid_stream_results_summary(bad_warm, runtime.max_tokens)
                )
            health_start = wait_drained(
                port,
                setup_deadline,
                f"steady-state warmup {steady_state_warmup_waves}",
            )
            graph_warm = mixed.graph_snapshot(health_start)
            batching_warm = mixed.batching_snapshot(health_start)
            if batching_warm["prefix_cache_enabled"] != prefix_cache_enabled:
                warm_failures = [
                    "prefix cache enabled capability changed during steady warmup"
                ]
            else:
                warm_failures = []
            prefix_warm = prefix_cache_snapshot(health_start)
            debug_warm = mixed.json_request(port, "GET", "/v1/debug/model-state")
            batched_warm = batched_state_cache_snapshot(debug_warm, runtime)
            resident_warm = resident_recurrent_state_snapshot(debug_warm, runtime)
            warm_failures.extend(
                mixed.attest_runtime(
                    runtime.variant_id,
                    health_start,
                    debug_warm,
                    rocm_graph_cache_entries=runtime.graph_cache_max,
                )
            )
            if batched_warm is not None and batched_warm["active_leases"] != 0:
                warm_failures.append("steady warmup retained a batched-state lease")
            warm_failures.extend(
                resident_recurrent_state_drain_failures(
                    resident_warm, "steady warmup"
                )
            )
            if graph_warm["failures"] != 0 or graph_warm["fallback_total"] != 0:
                warm_failures.append("graph failure or fallback occurred during steady warmup")
            leaked_warm = unaccounted_blocks(batching_warm, prefix_warm)
            if leaked_warm != 0:
                warm_failures.append(
                    f"steady warmup retained {leaked_warm} blocks outside the prefix cache"
                )
            if prefix_warm["active_leases"] != 0:
                warm_failures.append("steady warmup retained active prefix-cache leases")
            if prefix_warm["pending_release_entries"] != 0:
                warm_failures.append("steady warmup retained pending prefix releases")
            # Commit evidence only after the whole response cohort has drained.
            # A policy failure discovered in that drained snapshot must not erase
            # the requests that actually completed.
            steady_state_warmup_requests, steady_state_warmup_waves = (
                record_drained_warmup_wave(
                    steady_state_warmup_requests,
                    steady_state_warmup_waves,
                    warm_results,
                )
            )
            if warm_failures:
                raise SoakError("; ".join(warm_failures))
            mixed.trace(
                "soak_steady_state_warmup",
                cached_entries=prefix_warm["cached_entries"],
                max_entries=prefix_warm["max_entries"],
                requests=steady_state_warmup_requests,
                waves=steady_state_warmup_waves,
            )

        previous_gpu = sampler.read_bytes()
        stabilization_gpu_high_water = previous_gpu
        previous_memory = process_memory_snapshot(process.pid)
        stabilization_memory_baseline = previous_memory
        previous_process_mappings = (
            process_memory_mapping_snapshot(process.pid)
            if runtime.backend == "vulkan"
            else None
        )
        observed_process_mappings_start = previous_process_mappings
        observed_process_mappings_end = previous_process_mappings
        previous_vulkan_buffers = vulkan_buffer_snapshot(health_start, runtime)
        observed_vulkan_buffers_start = previous_vulkan_buffers
        observed_vulkan_buffers_end = previous_vulkan_buffers
        previous_vulkan_pool = vulkan_buffer_pool_snapshot(health_start, runtime)
        observed_vulkan_pool_start = previous_vulkan_pool
        observed_vulkan_pool_end = previous_vulkan_pool
        debug_stabilization_start = mixed.json_request(
            port, "GET", "/v1/debug/model-state"
        )
        previous_batched_state = batched_state_cache_snapshot(
            debug_stabilization_start, runtime
        )
        resident_stabilization_start = resident_recurrent_state_snapshot(
            debug_stabilization_start, runtime
        )
        failures.extend(
            resident_recurrent_state_drain_failures(
                resident_stabilization_start, "stabilization baseline"
            )
        )
        observed_batched_state_start = previous_batched_state
        observed_batched_state_end = previous_batched_state
        observed_stabilization_batching_start = batching_warm
        observed_stabilization_batching_end = batching_warm
        stabilization_started = time.monotonic()
        while stabilization_cycles < runtime.max_stabilization_cycles:
            cycle_failures: list[str] = []
            for offset in range(len(runtime.wave_concurrency)):
                stabilization_wave = (
                    stabilization_cycles * len(runtime.wave_concurrency) + offset
                )
                stable_results = run_wave(
                    port,
                    wave=stabilization_wave,
                    base_seed=seed,
                    deadline=setup_deadline,
                    phase="stabilize",
                    runtime=runtime,
                    worker_evidence=worker_evidence,
                )
                stabilization_requests += len(stable_results)
                bad_stable = [
                    result
                    for result in stable_results
                    if not valid_stream_result(result, runtime.max_tokens)
                ]
                if bad_stable:
                    cycle_failures.append(
                        "stabilization produced invalid responses: "
                        + invalid_stream_results_summary(
                            bad_stable, runtime.max_tokens
                        )
                    )
                if (stabilization_wave + 1) % runtime.cancel_every_waves == 0:
                    cancellation_failure = run_cancellation(
                        port,
                        wave=stabilization_wave,
                        base_seed=seed + 2_000_000,
                        phase="stabilize",
                        deadline=setup_deadline,
                        runtime=runtime,
                    )
                    if cancellation_failure is None:
                        stabilization_cancellations += 1
                    else:
                        cycle_failures.append(cancellation_failure)

                health_start = wait_drained(
                    port,
                    setup_deadline,
                    f"stabilization wave {stabilization_wave}",
                )
                graph_stable = mixed.graph_snapshot(health_start)
                batching_stable = mixed.batching_snapshot(health_start)
                observed_stabilization_batching_end = batching_stable
                prefix_stable = prefix_cache_snapshot(health_start)
                if batching_stable["prefix_cache_enabled"] != prefix_cache_enabled:
                    cycle_failures.append(
                        "prefix cache enabled capability changed during stabilization"
                    )
                debug_stable = mixed.json_request(
                    port, "GET", "/v1/debug/model-state"
                )
                current_batched_state = batched_state_cache_snapshot(
                    debug_stable, runtime
                )
                current_resident_state = resident_recurrent_state_snapshot(
                    debug_stable, runtime
                )
                observed_batched_state_end = current_batched_state
                cycle_failures.extend(
                    mixed.attest_runtime(
                        runtime.variant_id,
                        health_start,
                        debug_stable,
                        rocm_graph_cache_entries=runtime.graph_cache_max,
                    )
                )
                if (
                    current_batched_state is not None
                    and current_batched_state["active_leases"] != 0
                ):
                    cycle_failures.append(
                        "stabilization retained a batched-state lease after drain"
                    )
                cycle_failures.extend(
                    resident_recurrent_state_drain_failures(
                        current_resident_state,
                        f"stabilization wave {stabilization_wave}",
                    )
                )
                if (
                    graph_stable["failures"] != 0
                    or graph_stable["fallback_total"] != 0
                ):
                    cycle_failures.append(
                        "graph failure or fallback occurred during stabilization"
                    )
                if (
                    graph_stable["active_graph_slot_count"] != 0
                    or graph_stable["tracked_decode_owner_count"] != 0
                ):
                    cycle_failures.append(
                        "stabilization retained an active graph slot or timeline"
                    )
                if (
                    graph_stable["captured_graph_count"] > runtime.graph_cache_max
                    or graph_stable["graph_slot_count"] > runtime.graph_cache_max
                ):
                    cycle_failures.append("stabilization exceeded the graph cache bound")
                if unaccounted_blocks(batching_stable, prefix_stable) != 0:
                    cycle_failures.append(
                        "stabilization retained blocks outside the prefix cache"
                    )
                if (
                    prefix_stable["active_leases"] != 0
                    or prefix_stable["pending_release_entries"] != 0
                ):
                    cycle_failures.append(
                        "stabilization retained active or pending prefix ownership"
                    )
                if prefix_cache_enabled:
                    if (
                        prefix_stable["cached_entries"]
                        != prefix_stable["max_entries"]
                        or prefix_stable["cached_state_bytes"]
                        != prefix_stable["max_state_bytes"]
                    ):
                        cycle_failures.append(
                            "stabilization lost full prefix-cache residency"
                        )
                else:
                    cycle_failures.extend(
                        disabled_prefix_cache_failures(
                            prefix_stable, phase="stabilization"
                        )
                    )
                if process.poll() is not None:
                    raise SoakError(
                        "server exited during stabilization "
                        f"({process.returncode})"
                    )
                if any(
                    event.category == "device_fault"
                    for event in server_log.events_since(stabilization_started)
                ):
                    cycle_failures.append(
                        "server logged a device fault during stabilization"
                    )

            if cycle_failures:
                raise SoakError("; ".join(dict.fromkeys(cycle_failures)))
            current_gpu = sampler.read_bytes()
            current_memory = process_memory_snapshot(process.pid)
            current_process_mappings = (
                process_memory_mapping_snapshot(process.pid)
                if runtime.backend == "vulkan"
                else None
            )
            observed_process_mappings_end = current_process_mappings
            current_vulkan_buffers = vulkan_buffer_snapshot(health_start, runtime)
            observed_vulkan_buffers_end = current_vulkan_buffers
            current_vulkan_pool = vulkan_buffer_pool_snapshot(health_start, runtime)
            observed_vulkan_pool_end = current_vulkan_pool
            if (
                previous_vulkan_buffers is not None
                and current_vulkan_buffers is not None
            ):
                accounting_failures = vulkan_buffer_accounting_failures(
                    previous_vulkan_buffers, current_vulkan_buffers
                )
                if accounting_failures:
                    raise SoakError("; ".join(accounting_failures))
            adjacent_gpu_delta = max(0, current_gpu - previous_gpu)
            gpu_delta = stabilization_gpu_growth_delta(
                runtime,
                current_gpu=current_gpu,
                previous_gpu=previous_gpu,
                stabilization_gpu_high_water=stabilization_gpu_high_water,
            )
            stabilization_gpu_high_water = max(
                stabilization_gpu_high_water, current_gpu
            )
            rss_delta = max(0, current_memory.rss_bytes - previous_memory.rss_bytes)
            stabilization_final_gpu_delta = gpu_delta
            stabilization_final_rss_delta = rss_delta
            stabilization_max_gpu_delta = max(stabilization_max_gpu_delta, gpu_delta)
            stabilization_max_rss_delta = max(stabilization_max_rss_delta, rss_delta)
            stabilization_rss_growth = max(
                0, current_memory.rss_bytes - stabilization_memory_baseline.rss_bytes
            )
            vulkan_live_bytes_delta = 0
            vulkan_allocation_count = 0
            vulkan_free_count = 0
            if (
                previous_vulkan_buffers is not None
                and current_vulkan_buffers is not None
            ):
                vulkan_live_bytes_delta = max(
                    0,
                    vulkan_buffer_live_bytes(current_vulkan_buffers)
                    - vulkan_buffer_live_bytes(previous_vulkan_buffers),
                )
                if vulkan_live_bytes_delta > 0:
                    stabilization_vulkan_growth_cycles += 1
                vulkan_allocation_count = vulkan_buffer_counter_delta(
                    previous_vulkan_buffers, current_vulkan_buffers, "allocations"
                )
                vulkan_free_count = vulkan_buffer_counter_delta(
                    previous_vulkan_buffers, current_vulkan_buffers, "frees"
                )
            vulkan_pool_cache_miss_count = 0
            vulkan_pool_eviction_count = 0
            vulkan_pool_uncached_allocation_count = 0
            if previous_vulkan_pool is not None and current_vulkan_pool is not None:
                vulkan_pool_cache_miss_count = vulkan_buffer_pool_counter_delta(
                    previous_vulkan_pool, current_vulkan_pool, "cache_misses"
                )
                vulkan_pool_eviction_count = vulkan_buffer_pool_counter_delta(
                    previous_vulkan_pool, current_vulkan_pool, "eviction_count"
                )
                vulkan_pool_uncached_allocation_count = (
                    vulkan_buffer_pool_counter_delta(
                        previous_vulkan_pool,
                        current_vulkan_pool,
                        "uncached_allocation_count",
                    )
                )
            stable_cycle = stabilization_cycle_is_stable(
                runtime,
                gpu_delta=gpu_delta,
                rss_delta=rss_delta,
                vulkan_live_bytes_delta=vulkan_live_bytes_delta,
                vulkan_allocation_count=vulkan_allocation_count,
                vulkan_free_count=vulkan_free_count,
                vulkan_pool_cache_miss_count=vulkan_pool_cache_miss_count,
                vulkan_pool_eviction_count=vulkan_pool_eviction_count,
                vulkan_pool_uncached_allocation_count=(
                    vulkan_pool_uncached_allocation_count
                ),
            )
            if stable_cycle:
                stabilization_stable_cycles += 1
            else:
                stabilization_stable_cycles = 0
            stabilization_cycles += 1
            cycle_trace = {
                "cycle": stabilization_cycles,
                "adjacent_gpu_delta_bytes": adjacent_gpu_delta,
                "gpu_delta_bytes": gpu_delta,
                "gpu_memory_bytes": current_gpu,
                "gpu_memory_high_water_bytes": stabilization_gpu_high_water,
                "rss_anon_delta_bytes": max(
                    0, current_memory.rss_anon_bytes - previous_memory.rss_anon_bytes
                ),
                "rss_delta_bytes": rss_delta,
                "rss_growth_bytes": stabilization_rss_growth,
                "rss_file_delta_bytes": max(
                    0, current_memory.rss_file_bytes - previous_memory.rss_file_bytes
                ),
                "rss_shmem_delta_bytes": max(
                    0,
                    current_memory.rss_shmem_bytes - previous_memory.rss_shmem_bytes,
                ),
                "stable_cycles": stabilization_stable_cycles,
                "swap_delta_bytes": max(
                    0, current_memory.swap_bytes - previous_memory.swap_bytes
                ),
            }
            if (
                previous_vulkan_buffers is not None
                and current_vulkan_buffers is not None
            ):
                cycle_trace.update(
                    vulkan_allocated_bytes=vulkan_buffer_counter_delta(
                        previous_vulkan_buffers,
                        current_vulkan_buffers,
                        "allocated_bytes",
                    ),
                    vulkan_allocation_count=vulkan_allocation_count,
                    vulkan_free_count=vulkan_free_count,
                    vulkan_freed_bytes=vulkan_buffer_counter_delta(
                        previous_vulkan_buffers,
                        current_vulkan_buffers,
                        "freed_bytes",
                    ),
                    vulkan_live_device_local_bytes=current_vulkan_buffers[
                        "live_device_local_bytes"
                    ],
                    vulkan_live_host_visible_bytes=current_vulkan_buffers[
                        "live_host_visible_bytes"
                    ],
                    vulkan_live_bytes_delta=vulkan_live_bytes_delta,
                )
            if previous_vulkan_pool is not None and current_vulkan_pool is not None:
                for field in VULKAN_BUFFER_POOL_COUNTER_FIELDS:
                    vulkan_buffer_pool_counter_delta(
                        previous_vulkan_pool, current_vulkan_pool, field
                    )
                cycle_trace.update(
                    vulkan_buffer_pool_miss_trace_values(
                        previous_vulkan_pool, current_vulkan_pool
                    )
                )
                cycle_trace.update(
                    vulkan_pool_borrowed_bytes=current_vulkan_pool["borrowed_bytes"],
                    vulkan_pool_evicted_bytes=vulkan_buffer_pool_counter_delta(
                        previous_vulkan_pool, current_vulkan_pool, "evicted_bytes"
                    ),
                    vulkan_pool_eviction_count=vulkan_pool_eviction_count,
                    vulkan_pool_free_bytes=current_vulkan_pool["free_bytes"],
                    vulkan_pool_retained_bytes=current_vulkan_pool["retained_bytes"],
                    vulkan_pool_retained_bytes_delta=max(
                        0,
                        current_vulkan_pool["retained_bytes"]
                        - previous_vulkan_pool["retained_bytes"],
                    ),
                    vulkan_pool_uncached_allocated_bytes=vulkan_buffer_pool_counter_delta(
                        previous_vulkan_pool,
                        current_vulkan_pool,
                        "uncached_allocated_bytes",
                    ),
                    vulkan_pool_uncached_allocation_count=(
                        vulkan_pool_uncached_allocation_count
                    ),
                )
            if previous_batched_state is not None and current_batched_state is not None:
                cycle_trace.update(
                    batched_state_cache_trace_values(
                        previous_batched_state, current_batched_state
                    )
                )
            if (
                previous_process_mappings is not None
                and current_process_mappings is not None
            ):
                cycle_trace.update(
                    process_memory_mapping_trace(
                        previous_process_mappings, current_process_mappings
                    )
                )
            mixed.trace("soak_stabilization_cycle", **cycle_trace)
            previous_gpu = current_gpu
            previous_memory = current_memory
            previous_vulkan_buffers = current_vulkan_buffers
            previous_vulkan_pool = current_vulkan_pool
            previous_batched_state = current_batched_state
            previous_process_mappings = current_process_mappings
            if (
                runtime.backend == "vulkan"
                and stabilization_rss_growth > memory_growth_limit_bytes
            ):
                raise SoakError(
                    "Vulkan stabilization RSS growth exceeded the cumulative limit: "
                    f"{stabilization_rss_growth} > {memory_growth_limit_bytes} bytes"
                )
            if (
                stabilization_cycles >= runtime.min_stabilization_cycles
                and stabilization_stable_cycles >= runtime.required_stable_cycles
            ):
                break
        else:
            boundary = "GPU/owned-buffer" if runtime.backend == "vulkan" else "GPU/RSS"
            raise SoakError(
                f"{boundary} memory did not stabilize within "
                f"{runtime.max_stabilization_cycles} cycles"
            )

        graph_start = mixed.graph_snapshot(health_start)
        graph_end = graph_start
        batching_start = mixed.batching_snapshot(health_start)
        batching_end = batching_start
        if batching_start["prefix_cache_enabled"] != prefix_cache_enabled:
            raise SoakError(
                "prefix cache enabled capability changed before measurement"
            )
        prefix_start = prefix_cache_snapshot(health_start)
        prefix_end = prefix_start
        if not prefix_cache_enabled:
            disabled_failures = disabled_prefix_cache_failures(
                prefix_start, phase="measurement baseline"
            )
            if disabled_failures:
                raise SoakError("; ".join(disabled_failures))
        vulkan_buffers_start = vulkan_buffer_snapshot(health_start, runtime)
        vulkan_buffers_end = vulkan_buffers_start
        vulkan_pool_start = vulkan_buffer_pool_snapshot(health_start, runtime)
        vulkan_pool_end = vulkan_pool_start
        debug_measurement_start = mixed.json_request(
            port, "GET", "/v1/debug/model-state"
        )
        batched_state_start = batched_state_cache_snapshot(
            debug_measurement_start, runtime
        )
        batched_state_end = batched_state_start
        resident_state_start = resident_recurrent_state_snapshot(
            debug_measurement_start, runtime
        )
        resident_state_end = resident_state_start
        baseline_resident_failures = resident_recurrent_state_drain_failures(
            resident_state_start, "measurement baseline"
        )
        if baseline_resident_failures:
            raise SoakError("; ".join(baseline_resident_failures))
        gpu_start = measurement_gpu_baseline(
            runtime,
            current_gpu=sampler.read_bytes(),
            stabilization_gpu_high_water=stabilization_gpu_high_water,
        )
        gpu_end = gpu_start
        rss_start = process_memory_snapshot(process.pid).rss_bytes
        rss_end = rss_start
        sampler.start()
        measurement_started = time.monotonic()
        measurement_deadline = measurement_phase_deadline(
            measurement_started, minimum_duration_seconds, runtime
        )
        mixed.trace(
            "soak_measurement_started",
            deadline_seconds=minimum_duration_seconds + runtime.request_timeout_seconds,
            minimum_duration_seconds=minimum_duration_seconds,
            setup_elapsed_seconds=measurement_started - runtime_started,
        )
        rss_samples = [rss_start]

        while True:
            elapsed_before_wave = phase_elapsed_seconds(measurement_started)
            if wave > 0 and elapsed_before_wave >= minimum_duration_seconds:
                break
            wave_failures: list[str] = []
            wave_results = run_wave(
                port,
                wave=wave,
                base_seed=seed,
                deadline=measurement_deadline,
                runtime=runtime,
                worker_evidence=worker_evidence,
            )
            bad = [
                result
                for result in wave_results
                if not valid_stream_result(result, runtime.max_tokens)
            ]
            if bad:
                wave_failures.append(
                    "wave produced invalid responses: "
                    + invalid_stream_results_summary(bad, runtime.max_tokens)
                )

            if (wave + 1) % runtime.cancel_every_waves == 0:
                cancellation_failure = run_cancellation(
                    port,
                    wave=wave,
                    base_seed=seed,
                    phase="measured",
                    deadline=measurement_deadline,
                    runtime=runtime,
                )
                if cancellation_failure is not None:
                    wave_failures.append(cancellation_failure)
                else:
                    cancellations += 1

            health = wait_drained(
                port, measurement_deadline, f"soak wave {wave}"
            )
            debug = mixed.json_request(port, "GET", "/v1/debug/model-state")
            batched_state = batched_state_cache_snapshot(debug, runtime)
            resident_state = resident_recurrent_state_snapshot(debug, runtime)
            observed_batched_state_end = batched_state
            wave_failures.extend(
                mixed.attest_runtime(
                    runtime.variant_id,
                    health,
                    debug,
                    rocm_graph_cache_entries=runtime.graph_cache_max,
                )
            )
            graph = mixed.graph_snapshot(health)
            batching = mixed.batching_snapshot(health)
            prefix = prefix_cache_snapshot(health)
            health_end = health
            graph_end = graph
            batching_end = batching
            prefix_end = prefix
            batched_state_end = batched_state
            resident_state_end = resident_state
            if batching["prefix_cache_enabled"] != prefix_cache_enabled:
                wave_failures.append(
                    f"prefix cache enabled capability changed in wave {wave}"
                )
            if not prefix_cache_enabled:
                wave_failures.extend(
                    disabled_prefix_cache_failures(
                        prefix, phase=f"measured wave {wave}"
                    )
                )
            if graph["failures"] != graph_start["failures"]:
                wave_failures.append(f"graph failure counter changed in wave {wave}")
            if (
                graph["active_graph_slot_count"] != 0
                or graph["tracked_decode_owner_count"] != 0
            ):
                wave_failures.append(
                    f"wave {wave} retained an active graph slot or timeline"
                )
            if (
                graph["captured_graph_count"] > runtime.graph_cache_max
                or graph["graph_slot_count"] > runtime.graph_cache_max
            ):
                wave_failures.append(f"wave {wave} exceeded the graph cache bound")
            leaked_blocks = unaccounted_blocks(batching, prefix)
            if leaked_blocks != 0:
                wave_failures.append(
                    f"wave {wave} retained {leaked_blocks} KV blocks outside the prefix cache"
                )
            if prefix["active_leases"] != 0:
                wave_failures.append(
                    f"wave {wave} retained {prefix['active_leases']} prefix-cache leases"
                )
            if prefix["pending_release_entries"] != 0:
                wave_failures.append(
                    f"wave {wave} retained {prefix['pending_release_entries']} "
                    "pending prefix releases"
                )
            if batched_state is not None and batched_state["active_leases"] != 0:
                wave_failures.append(
                    f"wave {wave} retained a batched-state lease after drain"
                )
            wave_failures.extend(
                resident_recurrent_state_drain_failures(
                    resident_state, f"measured wave {wave}"
                )
            )
            if batched_state_start is not None and batched_state is not None:
                concurrent_misses = batched_state_cache_counter_delta(
                    batched_state_start,
                    batched_state,
                    "take_miss_while_leased_count",
                )
                replacement_evictions = batched_state_cache_counter_delta(
                    batched_state_start,
                    batched_state,
                    "park_replacement_eviction_count",
                )
                if concurrent_misses != 0 or replacement_evictions != 0:
                    wave_failures.append(
                        "batched recurrent-state ownership overlapped after "
                        f"stabilization: misses_while_leased={concurrent_misses}, "
                        f"park_replacements={replacement_evictions}"
                    )
            if process.poll() is not None:
                raise SoakError(f"server exited during wave {wave} ({process.returncode})")
            device_faults = [
                event
                for event in server_log.events_since(measurement_started)
                if event.category == "device_fault"
            ]
            if device_faults:
                wave_failures.append(
                    f"server logged a device fault during wave {wave}: "
                    f"{device_faults[-1].message}"
                )

            current_gpu = sampler.read_bytes()
            current_memory = process_memory_snapshot(process.pid)
            current_vulkan_buffers = vulkan_buffer_snapshot(health, runtime)
            current_vulkan_pool = vulkan_buffer_pool_snapshot(health, runtime)
            current_rss = current_memory.rss_bytes
            gpu_end = current_gpu
            rss_end = current_rss
            vulkan_buffers_end = current_vulkan_buffers
            vulkan_pool_end = current_vulkan_pool
            rss_samples.append(current_rss)
            if current_gpu > gpu_start + memory_growth_limit_bytes:
                wave_failures.append(
                    f"GPU memory grew by {current_gpu - gpu_start} bytes after warmup"
                )
            if current_rss > rss_start + memory_growth_limit_bytes:
                wave_failures.append(
                    f"RSS grew by {current_rss - rss_start} bytes after warmup"
                )
            wave_trace = {
                "cancellations": cancellations,
                "elapsed_seconds": time.monotonic() - measurement_started,
                "gpu_memory_bytes": current_gpu,
                "requests": len(wave_results),
                "rss_anon_bytes": current_memory.rss_anon_bytes,
                "rss_bytes": current_rss,
                "rss_file_bytes": current_memory.rss_file_bytes,
                "rss_shmem_bytes": current_memory.rss_shmem_bytes,
                "swap_bytes": current_memory.swap_bytes,
                "wave": wave,
            }
            if (
                vulkan_buffers_start is not None
                and current_vulkan_buffers is not None
            ):
                vulkan_allocation_growth = vulkan_buffer_counter_delta(
                    vulkan_buffers_start,
                    current_vulkan_buffers,
                    "allocations",
                )
                wave_trace.update(
                    vulkan_allocated_bytes=vulkan_buffer_counter_delta(
                        vulkan_buffers_start,
                        current_vulkan_buffers,
                        "allocated_bytes",
                    ),
                    vulkan_allocation_count=vulkan_allocation_growth,
                    vulkan_free_count=vulkan_buffer_counter_delta(
                        vulkan_buffers_start, current_vulkan_buffers, "frees"
                    ),
                    vulkan_freed_bytes=vulkan_buffer_counter_delta(
                        vulkan_buffers_start,
                        current_vulkan_buffers,
                        "freed_bytes",
                    ),
                    vulkan_live_device_local_bytes=current_vulkan_buffers[
                        "live_device_local_bytes"
                    ],
                    vulkan_live_host_visible_bytes=current_vulkan_buffers[
                        "live_host_visible_bytes"
                    ],
                )
                if (
                    runtime.vulkan_allocation_growth_limit_count is not None
                    and vulkan_allocation_growth
                    > runtime.vulkan_allocation_growth_limit_count
                ):
                    wave_failures.append(
                        "Vulkan buffer allocations continued after stabilization: "
                        f"{vulkan_allocation_growth} > "
                        f"{runtime.vulkan_allocation_growth_limit_count}"
                    )
            if batched_state_start is not None and batched_state is not None:
                wave_trace.update(
                    batched_state_cache_trace_values(
                        batched_state_start, batched_state
                    )
                )
            if vulkan_pool_start is not None and current_vulkan_pool is not None:
                wave_trace.update(
                    vulkan_buffer_pool_miss_trace_values(
                        vulkan_pool_start, current_vulkan_pool
                    )
                )
            all_results.extend(wave_results)
            mixed.trace("soak_wave_complete", **wave_trace)
            wave += 1
            if wave_failures:
                failures.extend(wave_failures)
                break

        sampler.stop_sampling()
        accelerator_sampler.close()
        health_end = wait_drained(
            port, measurement_deadline, "soak final health"
        )
        graph_end = mixed.graph_snapshot(health_end)
        batching_end = mixed.batching_snapshot(health_end)
        if batching_end["prefix_cache_enabled"] != prefix_cache_enabled:
            failures.append("prefix cache enabled capability changed before final drain")
        prefix_end = prefix_cache_snapshot(health_end)
        vulkan_buffers_end = vulkan_buffer_snapshot(health_end, runtime)
        vulkan_pool_end = vulkan_buffer_pool_snapshot(health_end, runtime)
        debug_end = mixed.json_request(port, "GET", "/v1/debug/model-state")
        batched_state_end = batched_state_cache_snapshot(debug_end, runtime)
        resident_state_end = resident_recurrent_state_snapshot(debug_end, runtime)
        failures.extend(
            resident_recurrent_state_drain_failures(resident_state_end, "final drain")
        )
        observed_batched_state_end = batched_state_end
        observed_process_mappings_end = (
            process_memory_mapping_snapshot(process.pid)
            if runtime.backend == "vulkan"
            else None
        )
        failures.extend(
            mixed.attest_runtime_execution(runtime.variant_id, health_start, health_end)
        )
        events = server_log.events_since(measurement_started)
        all_server_events = server_log.events_since(runtime_started)
        gpu_end = sampler.read_bytes()
        rss_end = process_memory_snapshot(process.pid).rss_bytes
        if host_guard is not None:
            host_guard.close()
        gpu_peak = max([gpu_start, gpu_end, *sampler.samples])
        rss_peak = max([rss_start, rss_end, *rss_samples])
        sync_values = mixed.external_yield_sync_metric_values(health_start, health_end)
        # Fallback stats are flattened by graph_snapshot with a `fallback_` prefix.
        fallback_delta = mixed.counter_delta(graph_start, graph_end, "fallback_total")
        duration = phase_elapsed_seconds(measurement_started)
        result_evidence = measurement_result_evidence(
            warmup_itl_ms=warmup.itl_ms,
            results=all_results,
            measurement_events=events,
            all_server_events=all_server_events,
            expected_completion_tokens=runtime.max_tokens,
            cancellation_count=cancellations,
            duration_seconds=duration,
            wave_count=wave,
            steady_state_warmup_request_count=steady_state_warmup_requests,
            steady_state_warmup_wave_count=steady_state_warmup_waves,
        )
        successes = result_evidence.successes
        attributed = result_evidence.attributed_itl_outliers
        unexplained = result_evidence.unexplained_itl_outliers
        request_failures = int(result_evidence.values["request_failure_count"])
        zero_tokens = int(result_evidence.values["zero_token_response_count"])
        values = {
            **result_evidence.values,
            **accelerator_sampler.metric_values_since(measurement_started),
            "batching_error_count": mixed.counter_delta(
                batching_start, batching_end, "total_errors"
            ),
            "batching_max_observed_active_requests": batching_end[
                "max_observed_active_requests"
            ],
            "batching_max_observed_batch_size": batching_end["max_observed_batch_size"],
            "external_yield_sync_failure_count": sync_values[
                "external_yield_sync_failure_count"
            ],
            "external_yield_sync_max_ms": sync_values["external_yield_sync_max_ms"],
            "external_yield_sync_slow_count": sync_values[
                "external_yield_sync_slow_count"
            ],
            "graph_capture_failure_count": mixed.counter_delta(
                graph_start, graph_end, "capture_failures"
            ),
            "graph_capture_success_count": mixed.counter_delta(
                graph_start, graph_end, "capture_successes"
            ),
            "graph_fallback_count": fallback_delta,
            "graph_retained_count_end": graph_end["captured_graph_count"],
            "graph_retained_count_start": graph_start["captured_graph_count"],
            "graph_replay_failure_count": mixed.counter_delta(
                graph_start, graph_end, "replay_failures"
            ),
            "graph_replay_success_count": mixed.counter_delta(
                graph_start, graph_end, "replay_successes"
            ),
            "graph_slot_active_count_end": graph_end["active_graph_slot_count"],
            "graph_slot_count_end": graph_end["graph_slot_count"],
            "graph_slot_count_start": graph_start["graph_slot_count"],
            "graph_slot_create_count": mixed.counter_delta(
                graph_start, graph_end, "graph_slot_create_count"
            ),
            "graph_slot_idle_count_end": graph_end["idle_graph_slot_count"],
            "graph_slot_reuse_count": mixed.counter_delta(
                graph_start, graph_end, "graph_slot_reuse_count"
            ),
            "gpu_memory_baseline_bytes": gpu_start,
            "gpu_memory_end_bytes": gpu_end,
            "gpu_memory_growth_bytes": max(0, gpu_end - gpu_start),
            "gpu_memory_peak_bytes": gpu_peak,
            "gpu_memory_peak_growth_bytes": max(0, gpu_peak - gpu_start),
            "kv_blocks_end": batching_end["blocks_total"],
            "kv_blocks_start": batching_start["blocks_total"],
            "kv_blocks_used_end": batching_end["blocks_used"],
            "kv_unaccounted_blocks_end": unaccounted_blocks(
                batching_end, prefix_end
            ),
            "prefix_cache_active_leases_end": prefix_end["active_leases"],
            "prefix_cache_baseline_cached_entries": prefix_start["cached_entries"],
            "prefix_cache_baseline_state_bytes": prefix_start["cached_state_bytes"],
            "prefix_cache_cached_blocks_end": prefix_end["cached_blocks"],
            "prefix_cache_cached_entries_end": prefix_end["cached_entries"],
            "prefix_cache_hit_blocks": mixed.counter_delta(
                prefix_start, prefix_end, "hit_blocks"
            ),
            "prefix_cache_hit_tokens": mixed.counter_delta(
                prefix_start, prefix_end, "hit_tokens"
            ),
            "prefix_cache_lookup_hit_count": mixed.counter_delta(
                prefix_start, prefix_end, "lookup_hits"
            ),
            "prefix_cache_lookup_miss_count": mixed.counter_delta(
                prefix_start, prefix_end, "lookup_misses"
            ),
            "prefix_cache_pending_release_entries_end": prefix_end[
                "pending_release_entries"
            ],
            "prefix_cache_state_bytes_end": prefix_end["cached_state_bytes"],
            "request_worker_residue_count": worker_evidence.peak_residue_count,
            "rss_baseline_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "rss_growth_bytes": max(0, rss_end - rss_start),
            "rss_peak_bytes": rss_peak,
            "measurement_final_snapshot_complete": 1,
            "shutdown_forced_count": 0,
            "shutdown_nonzero_count": 0,
            "snapshot_residue_count": 0,
            "stabilization_cancellation_count": stabilization_cancellations,
            "stabilization_cycle_count": stabilization_cycles,
            "stabilization_final_gpu_delta_bytes": stabilization_final_gpu_delta,
            "stabilization_final_rss_delta_bytes": stabilization_final_rss_delta,
            "stabilization_max_gpu_delta_bytes": stabilization_max_gpu_delta,
            "stabilization_max_rss_delta_bytes": stabilization_max_rss_delta,
            "stabilization_rss_growth_bytes": stabilization_rss_growth,
            "stabilization_request_count": stabilization_requests,
            "stabilization_stable_cycle_count": stabilization_stable_cycles,
        }
        if runtime.backend == "vulkan":
            assert observed_stabilization_batching_start is not None
            assert observed_stabilization_batching_end is not None
            stabilization_resident_values = (
                stabilization_resident_prefill_metric_values(
                    observed_stabilization_batching_start,
                    observed_stabilization_batching_end,
                )
            )
            values.update(stabilization_resident_values)
            stabilization_contract_values = {
                name.removeprefix("stabilization_"): value
                for name, value in stabilization_resident_values.items()
            }
            failures.extend(
                "stabilization " + failure
                for failure in resident_prefill_contract_failures(
                    stabilization_contract_values,
                    max_configured_rows=max(runtime.wave_concurrency),
                )
            )
            values["prefix_cache_enabled"] = prefix_cache_capability_value(
                batching_start, batching_end
            )
            if values["prefix_cache_enabled"] != 0:
                failures.append(
                    "Vulkan cross-request prefix reuse must remain correctness-quarantined"
                )
            values.update(
                resident_prefill_metric_values(batching_start, batching_end)
            )
            failures.extend(
                resident_prefill_contract_failures(
                    values,
                    max_configured_rows=max(runtime.wave_concurrency),
                )
            )
            assert resident_state_end is not None
            values.update(resident_recurrent_state_metric_values(resident_state_end))
        if vulkan_buffers_start is not None and vulkan_buffers_end is not None:
            vulkan_live_start = vulkan_buffer_live_bytes(vulkan_buffers_start)
            vulkan_live_end = vulkan_buffer_live_bytes(vulkan_buffers_end)
            values.update(
                vulkan_buffer_metric_values(vulkan_buffers_start, vulkan_buffers_end)
            )
            values["vulkan_buffer_stabilization_growth_cycle_count"] = (
                stabilization_vulkan_growth_cycles
            )
            failures.extend(
                vulkan_buffer_accounting_failures(
                    vulkan_buffers_start, vulkan_buffers_end
                )
            )
            if vulkan_live_end > vulkan_live_start:
                failures.append(
                    "live Vulkan buffer ownership grew after warmup: "
                    f"{vulkan_live_start} -> {vulkan_live_end} bytes"
                )
        if vulkan_pool_start is not None and vulkan_pool_end is not None:
            values.update(
                vulkan_buffer_pool_metric_values(vulkan_pool_start, vulkan_pool_end)
            )
            if vulkan_pool_end["retained_bytes"] > vulkan_pool_start["retained_bytes"]:
                failures.append(
                    "Vulkan buffer-pool retention grew after stabilization: "
                    f"{vulkan_pool_start['retained_bytes']} -> "
                    f"{vulkan_pool_end['retained_bytes']} bytes"
                )
        if (
            observed_batched_state_start is not None
            and observed_batched_state_end is not None
        ):
            values.update(
                batched_state_cache_metric_values(
                    observed_batched_state_start, observed_batched_state_end
                )
            )
            for name in (
                "batched_state_cache_active_leases_end",
                "batched_state_cache_completed_row_eviction_count",
                "batched_state_cache_explicit_invalidation_count",
                "batched_state_cache_explicit_invalidation_eviction_count",
                "batched_state_cache_park_replacement_eviction_count",
                "batched_state_cache_take_miss_while_leased_count",
            ):
                if values[name] != 0:
                    failures.append(f"{name}={values[name]}, expected 0")
            if values["batched_state_cache_max_active_leases"] > 1:
                failures.append(
                    "batched_state_cache_max_active_leases="
                    f"{values['batched_state_cache_max_active_leases']}, expected <= 1"
                )
            if values["batched_state_cache_entry_present_end"] != 1:
                failures.append("batched recurrent-state capacity was not parked at final drain")
            if values["batched_state_cache_resident_end"] != 1:
                failures.append("final batched recurrent-state capacity was not resident")
        if (
            observed_process_mappings_start is not None
            and observed_process_mappings_end is not None
        ):
            values.update(
                process_memory_mapping_metric_values(
                    observed_process_mappings_start,
                    observed_process_mappings_end,
                )
            )
        if host_guard is not None:
            values.update(host_guard.metric_values())
        if duration < minimum_duration_seconds:
            failures.append(
                "soak duration "
                f"{duration:.3f}s was below "
                f"{minimum_duration_seconds:.3f}s"
            )
        if request_failures != 0 or zero_tokens != 0:
            failures.append(
                f"soak had request_failures={request_failures}, zero_tokens={zero_tokens}"
            )
        if values["latency_phase_metadata_missing_count"] != 0:
            failures.append(
                f"{values['latency_phase_metadata_missing_count']} successful measured "
                "requests omitted terminal latency-phase metadata"
            )
        if unexplained != 0:
            failures.append(
                f"soak had {unexplained} unexplained ITL outliers "
                f"({attributed} additional outliers had bounded attribution)"
            )
        for name in (
            "graph_capture_failure_count",
            "graph_replay_failure_count",
            "graph_fallback_count",
            "graph_slot_active_count_end",
            "external_yield_sync_failure_count",
            "external_yield_sync_slow_count",
            "batching_error_count",
            "device_fault_event_count",
            "non_finite_response_count",
            "kv_unaccounted_blocks_end",
            "prefix_cache_active_leases_end",
            "prefix_cache_pending_release_entries_end",
        ):
            if values[name] != 0:
                failures.append(f"{name}={values[name]}, expected 0")
        if runtime.graph_execution_required:
            if values["graph_replay_success_count"] < 1:
                failures.append("soak completed without a measured graph replay")
            if values["graph_slot_reuse_count"] < 1:
                failures.append("soak completed without measured graph-slot reuse")
        elif any(
            values[name] != 0
            for name in (
                "graph_capture_success_count",
                "graph_replay_success_count",
                "graph_retained_count_end",
                "graph_retained_count_start",
                "graph_slot_count_end",
                "graph_slot_count_start",
                "graph_slot_create_count",
                "graph_slot_idle_count_end",
                "graph_slot_reuse_count",
            )
        ):
            failures.append("graph-disabled soak recorded graph execution activity")
        if values["graph_retained_count_end"] > runtime.graph_cache_max:
            failures.append("retained graph residency exceeded the graph cache bound")
        if values["graph_slot_count_end"] > runtime.graph_cache_max:
            failures.append("graph-slot residency exceeded the graph cache bound")
        if values["graph_slot_idle_count_end"] != values["graph_slot_count_end"]:
            failures.append("not every retained graph slot was idle at final drain")
        if prefix_cache_enabled:
            if values["prefix_cache_lookup_hit_count"] < 1:
                failures.append("soak completed without a measured prefix-cache hit")
            if values["prefix_cache_hit_blocks"] < 1:
                failures.append("soak completed without reusing a cached KV block")
            if prefix_start["cached_entries"] != prefix_start["max_entries"]:
                failures.append(
                    "measurement began before the prefix cache reached capacity"
                )
            if prefix_end["cached_entries"] != prefix_start["cached_entries"]:
                failures.append(
                    "prefix-cache entry residency changed during measured soak"
                )
            if prefix_end["cached_state_bytes"] != prefix_start["cached_state_bytes"]:
                failures.append(
                    "prefix-cache state residency changed during measured soak"
                )
        else:
            failures.extend(
                disabled_prefix_cache_failures(
                    prefix_start, phase="measurement baseline"
                )
            )
            failures.extend(
                disabled_prefix_cache_failures(prefix_end, phase="final drain")
            )
            for name in (
                "prefix_cache_baseline_cached_entries",
                "prefix_cache_baseline_state_bytes",
                "prefix_cache_cached_blocks_end",
                "prefix_cache_cached_entries_end",
                "prefix_cache_hit_blocks",
                "prefix_cache_hit_tokens",
                "prefix_cache_lookup_hit_count",
                "prefix_cache_lookup_miss_count",
                "prefix_cache_state_bytes_end",
            ):
                if values[name] != 0:
                    failures.append(f"{name}={values[name]} while prefix cache is disabled")
        if batching_end["blocks_total"] != batching_start["blocks_total"]:
            failures.append("KV block capacity changed during soak")
        if sampler.errors:
            failures.append("GPU memory sampler errors: " + ", ".join(sampler.errors))
        active_gpu_peak_growth_limit_bytes = (
            runtime.active_gpu_peak_growth_limit_bytes
            if runtime.active_gpu_peak_growth_limit_bytes is not None
            else memory_growth_limit_bytes
        )
        if gpu_peak > gpu_start + active_gpu_peak_growth_limit_bytes:
            failures.append(
                "peak GPU memory exceeded the active-workload growth limit: "
                f"{gpu_peak - gpu_start} > {active_gpu_peak_growth_limit_bytes} bytes"
            )
        if rss_peak > rss_start + memory_growth_limit_bytes:
            failures.append("peak RSS exceeded the post-warmup growth limit")
        if host_guard is not None:
            if host_guard.trip_reason is not None:
                failures.append(host_guard.trip_reason)
            if host_guard.errors:
                failures.append(
                    "host memory guard errors: " + ", ".join(host_guard.errors)
                )
            assert runtime.host_swap_growth_limit_bytes is not None
            if (
                values["host_swap_growth_bytes"]
                > runtime.host_swap_growth_limit_bytes
            ):
                failures.append(
                    "host swap growth exceeded the configured limit: "
                    f"{values['host_swap_growth_bytes']} > "
                    f"{runtime.host_swap_growth_limit_bytes} bytes"
                )
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        sampler.close()
        accelerator_sampler.close()
        if host_guard is not None:
            host_guard.close()
        shutdown = mixed.terminate_process(process)
        server_log.join()
        snapshot_residue = mixed.snapshot_payload_residue(snapshot_dir)
        shutil.rmtree(run_dir, ignore_errors=True)

    if sampler.errors:
        failures.append("GPU memory sampler errors: " + ", ".join(sampler.errors))

    if host_guard is not None:
        if host_guard.trip_reason is not None:
            failures.append(host_guard.trip_reason)
        if host_guard.errors:
            failures.append("host memory guard errors: " + ", ".join(host_guard.errors))
        assert runtime.host_swap_growth_limit_bytes is not None
        observed_host = host_guard.metric_values()
        if observed_host["host_swap_growth_bytes"] > runtime.host_swap_growth_limit_bytes:
            failures.append(
                "host swap growth exceeded the configured limit: "
                f"{observed_host['host_swap_growth_bytes']} > "
                f"{runtime.host_swap_growth_limit_bytes} bytes"
            )

    if values is None:
        values = {name: 0 for name in metric_definitions(runtime)}
        values["soak_duration_seconds"] = phase_elapsed_seconds(measurement_started)
        values["stabilization_cancellation_count"] = stabilization_cancellations
        values["stabilization_cycle_count"] = stabilization_cycles
        values["stabilization_final_gpu_delta_bytes"] = stabilization_final_gpu_delta
        values["stabilization_final_rss_delta_bytes"] = stabilization_final_rss_delta
        values["stabilization_max_gpu_delta_bytes"] = stabilization_max_gpu_delta
        values["stabilization_max_rss_delta_bytes"] = stabilization_max_rss_delta
        values["stabilization_rss_growth_bytes"] = stabilization_rss_growth
        values["stabilization_request_count"] = stabilization_requests
        values["stabilization_stable_cycle_count"] = stabilization_stable_cycles
        values["steady_state_warmup_request_count"] = steady_state_warmup_requests
        values["steady_state_warmup_wave_count"] = steady_state_warmup_waves
        values["request_worker_residue_count"] = worker_evidence.peak_residue_count
        if host_guard is not None:
            values.update(host_guard.metric_values())
        if (
            observed_vulkan_buffers_start is not None
            and observed_vulkan_buffers_end is not None
        ):
            values.update(
                vulkan_buffer_metric_values(
                    observed_vulkan_buffers_start, observed_vulkan_buffers_end
                )
            )
            values["vulkan_buffer_stabilization_growth_cycle_count"] = (
                stabilization_vulkan_growth_cycles
            )
        if (
            observed_vulkan_pool_start is not None
            and observed_vulkan_pool_end is not None
        ):
            values.update(
                vulkan_buffer_pool_metric_values(
                    observed_vulkan_pool_start, observed_vulkan_pool_end
                )
            )
        if (
            observed_batched_state_start is not None
            and observed_batched_state_end is not None
        ):
            values.update(
                batched_state_cache_metric_values(
                    observed_batched_state_start, observed_batched_state_end
                )
            )
        values.update(
            partial_stabilization_resident_prefill_metric_values(
                runtime,
                observed_stabilization_batching_start,
                observed_stabilization_batching_end,
            )
        )
        if (
            observed_process_mappings_start is not None
            and observed_process_mappings_end is not None
        ):
            values.update(
                process_memory_mapping_metric_values(
                    observed_process_mappings_start,
                    observed_process_mappings_end,
                )
            )
        if measurement_started is not None:
            assert warmup is not None
            measurement_events = server_log.events_since(measurement_started)
            result_evidence = measurement_result_evidence(
                warmup_itl_ms=warmup.itl_ms,
                results=all_results,
                measurement_events=measurement_events,
                all_server_events=server_log.events_since(runtime_started),
                expected_completion_tokens=runtime.max_tokens,
                cancellation_count=cancellations,
                duration_seconds=phase_elapsed_seconds(measurement_started),
                wave_count=wave,
                steady_state_warmup_request_count=steady_state_warmup_requests,
                steady_state_warmup_wave_count=steady_state_warmup_waves,
            )
            values.update(result_evidence.values)
            if gpu_start is not None and gpu_end is not None:
                gpu_peak = max([gpu_start, gpu_end, *sampler.samples])
                values.update(
                    {
                        "gpu_memory_baseline_bytes": gpu_start,
                        "gpu_memory_end_bytes": gpu_end,
                        "gpu_memory_growth_bytes": max(0, gpu_end - gpu_start),
                        "gpu_memory_peak_bytes": gpu_peak,
                        "gpu_memory_peak_growth_bytes": max(
                            0, gpu_peak - gpu_start
                        ),
                    }
                )
            if rss_start is not None and rss_end is not None:
                rss_peak = max([rss_start, rss_end, *rss_samples])
                values.update(
                    {
                        "rss_baseline_bytes": rss_start,
                        "rss_end_bytes": rss_end,
                        "rss_growth_bytes": max(0, rss_end - rss_start),
                        "rss_peak_bytes": rss_peak,
                    }
                )
            if graph_start is not None and graph_end is not None:
                values.update(
                    {
                        "graph_capture_failure_count": mixed.counter_delta(
                            graph_start, graph_end, "capture_failures"
                        ),
                        "graph_capture_success_count": mixed.counter_delta(
                            graph_start, graph_end, "capture_successes"
                        ),
                        "graph_fallback_count": mixed.counter_delta(
                            graph_start, graph_end, "fallback_total"
                        ),
                        "graph_replay_failure_count": mixed.counter_delta(
                            graph_start, graph_end, "replay_failures"
                        ),
                        "graph_replay_success_count": mixed.counter_delta(
                            graph_start, graph_end, "replay_successes"
                        ),
                        "graph_retained_count_start": graph_start[
                            "captured_graph_count"
                        ],
                        "graph_retained_count_end": graph_end[
                            "captured_graph_count"
                        ],
                        "graph_slot_active_count_end": graph_end[
                            "active_graph_slot_count"
                        ],
                        "graph_slot_count_start": graph_start["graph_slot_count"],
                        "graph_slot_count_end": graph_end["graph_slot_count"],
                        "graph_slot_create_count": mixed.counter_delta(
                            graph_start, graph_end, "graph_slot_create_count"
                        ),
                        "graph_slot_idle_count_end": graph_end[
                            "idle_graph_slot_count"
                        ],
                        "graph_slot_reuse_count": mixed.counter_delta(
                            graph_start, graph_end, "graph_slot_reuse_count"
                        ),
                    }
                )
            if batching_start is not None and batching_end is not None:
                values.update(
                    {
                        "batching_error_count": mixed.counter_delta(
                            batching_start, batching_end, "total_errors"
                        ),
                        "batching_max_observed_active_requests": batching_end[
                            "max_observed_active_requests"
                        ],
                        "batching_max_observed_batch_size": batching_end[
                            "max_observed_batch_size"
                        ],
                        "kv_blocks_start": batching_start["blocks_total"],
                        "kv_blocks_end": batching_end["blocks_total"],
                        "kv_blocks_used_end": batching_end["blocks_used"],
                    }
                )
                if runtime.backend == "vulkan":
                    values["prefix_cache_enabled"] = prefix_cache_capability_value(
                        batching_start, batching_end
                    )
                    values.update(
                        resident_prefill_metric_values(batching_start, batching_end)
                    )
            if prefix_start is not None and prefix_end is not None:
                values.update(
                    {
                        "kv_unaccounted_blocks_end": (
                            unaccounted_blocks(batching_end, prefix_end)
                            if batching_end is not None
                            else 0
                        ),
                        "prefix_cache_active_leases_end": prefix_end["active_leases"],
                        "prefix_cache_baseline_cached_entries": prefix_start[
                            "cached_entries"
                        ],
                        "prefix_cache_baseline_state_bytes": prefix_start[
                            "cached_state_bytes"
                        ],
                        "prefix_cache_cached_blocks_end": prefix_end["cached_blocks"],
                        "prefix_cache_cached_entries_end": prefix_end[
                            "cached_entries"
                        ],
                        "prefix_cache_hit_blocks": mixed.counter_delta(
                            prefix_start, prefix_end, "hit_blocks"
                        ),
                        "prefix_cache_hit_tokens": mixed.counter_delta(
                            prefix_start, prefix_end, "hit_tokens"
                        ),
                        "prefix_cache_lookup_hit_count": mixed.counter_delta(
                            prefix_start, prefix_end, "lookup_hits"
                        ),
                        "prefix_cache_lookup_miss_count": mixed.counter_delta(
                            prefix_start, prefix_end, "lookup_misses"
                        ),
                        "prefix_cache_pending_release_entries_end": prefix_end[
                            "pending_release_entries"
                        ],
                        "prefix_cache_state_bytes_end": prefix_end[
                            "cached_state_bytes"
                        ],
                    }
                )
            if health_start is not None and health_end is not None:
                values.update(
                    mixed.external_yield_sync_metric_values(health_start, health_end)
                )
            if vulkan_buffers_start is not None and vulkan_buffers_end is not None:
                values.update(
                    vulkan_buffer_metric_values(
                        vulkan_buffers_start, vulkan_buffers_end
                    )
                )
                values["vulkan_buffer_stabilization_growth_cycle_count"] = (
                    stabilization_vulkan_growth_cycles
                )
            if vulkan_pool_start is not None and vulkan_pool_end is not None:
                values.update(
                    vulkan_buffer_pool_metric_values(
                        vulkan_pool_start, vulkan_pool_end
                    )
                )
            if batched_state_start is not None and batched_state_end is not None:
                values.update(
                    batched_state_cache_metric_values(
                        batched_state_start, batched_state_end
                    )
                )
            if resident_state_end is not None:
                values.update(
                    resident_recurrent_state_metric_values(resident_state_end)
                )
    accelerator_values = accelerator_sampler.metric_values_since(
        measurement_started if measurement_started is not None else math.inf
    )
    values.update(accelerator_values)
    if accelerator_sampler.errors:
        failures.append(
            "accelerator telemetry errors: "
            + ", ".join(accelerator_sampler.errors)
        )
    if runtime.accelerator_telemetry_required:
        if accelerator_values["accelerator_telemetry_available"] != 1:
            failures.append("required accelerator telemetry was unavailable")
        if accelerator_values["accelerator_telemetry_sample_count"] < 1:
            failures.append(
                "required accelerator telemetry recorded no measurement samples"
            )
        if accelerator_values["accelerator_telemetry_active_sample_count"] < 1:
            failures.append(
                "required accelerator telemetry recorded no active GPU samples"
            )
    assert shutdown is not None
    values["shutdown_forced_count"] = int(shutdown.forced)
    values["shutdown_nonzero_count"] = int(shutdown.returncode != 0)
    values["snapshot_residue_count"] = len(snapshot_residue)
    if shutdown.forced:
        failures.append("server required forced shutdown")
    if shutdown.returncode != 0:
        failures.append(f"server shutdown returned {shutdown.returncode}")
    if snapshot_residue:
        failures.append("server left model snapshot residue: " + ", ".join(snapshot_residue))
    if values["request_worker_residue_count"] != 0:
        failures.append(
            "request_worker_residue_count="
            f"{values['request_worker_residue_count']}, expected 0"
        )
    details = " | ".join(dict.fromkeys(failures)) if failures else None
    return metrics_from_values(values, runtime), mixed.bounded_details(details)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--minimum-duration-seconds", required=True, type=float)
    parser.add_argument(
        "--memory-growth-limit-bytes",
        type=int,
        default=DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.minimum_duration_seconds) or args.minimum_duration_seconds < 60:
        parser.error("--minimum-duration-seconds must be finite and at least 60")
    if args.minimum_duration_seconds > 172800:
        parser.error("--minimum-duration-seconds must not exceed 172800")
    if args.memory_growth_limit_bytes < 0:
        parser.error("--memory-growth-limit-bytes must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    variant = os.environ.get(VARIANT_ENV, "")
    result_path_value = os.environ.get(RESULT_ENV)
    try:
        runtime = runtime_for_variant(variant)
    except SoakError as exc:
        print(
            str(exc),
            file=sys.stderr,
        )
        return 2
    if not result_path_value:
        print(f"{RESULT_ENV} is required", file=sys.stderr)
        return 2
    result_path = Path(result_path_value)
    status = "failed"
    details: str | None = None
    metrics = metrics_from_values(
        {name: 0 for name in metric_definitions(runtime)}, runtime
    )
    try:
        model_path = args.model_path.resolve(strict=True)
        if not model_path.is_dir():
            raise SoakError("--model-path must be a directory")
        metrics, details = execute(
            model_path,
            args.seed,
            args.minimum_duration_seconds,
            args.memory_growth_limit_bytes,
            runtime,
        )
        status = "passed" if details is None else "failed"
    except Exception as exc:
        details = mixed.bounded_details(f"{type(exc).__name__}: {exc}")
        mixed.trace("soak_qualification_error", details=details)
    result = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "status": status,
        "duration_seconds": time.monotonic() - started,
        "effective_config": effective_config(
            args.minimum_duration_seconds, args.memory_growth_limit_bytes, runtime
        ),
        "metrics": metrics,
        "tolerances": [],
        "details": details,
    }
    try:
        mixed.write_result(Path(result_path_value), result)
    except Exception as exc:
        print(f"cannot write soak qualification result: {exc}", file=sys.stderr)
        return 2
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
