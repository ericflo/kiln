#!/usr/bin/env python3
"""Prove Vulkan resident-prefill semantics across repeated changing cohorts."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import serve_development_soak as soak
import serve_mixed_load as mixed


ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "vulkan-resident-prefill-oracle"
VARIANT_ID = "vulkan-resident-prefill-oracle"
RESULT_ENV = mixed.RESULT_ENV
VARIANT_ENV = mixed.VARIANT_ENV
BUILD_TIMEOUT_SECONDS = 900.0
OVERALL_TIMEOUT_SECONDS = 1800.0
REQUEST_TIMEOUT_SECONDS = 600.0
WARMUP_MAX_TOKENS = 4
PROMPT_WORDS = 16
COHORT_MAX_TOKENS: tuple[tuple[int, ...], ...] = (
    (8, 12, 16, 20),
    (20, 16, 12, 8),
)
EXPECTED_REQUESTS = sum(len(cohort) for cohort in COHORT_MAX_TOKENS)
EXPECTED_COMPLETION_TOKENS = sum(sum(cohort) for cohort in COHORT_MAX_TOKENS)
HOST_MEMORY_FLOOR_BYTES = 8 * 1024 * 1024 * 1024
HOST_SWAP_GROWTH_LIMIT_BYTES = 512 * 1024 * 1024
GPU_PEAK_GROWTH_LIMIT_BYTES = 1024 * 1024 * 1024


def _effective_config() -> dict[str, Any]:
    value = mixed._variant_config(
        serving_profile="experimental",
        kv_autoscale_requested=False,
        kv_autoscale_enabled=False,
        memory_reclaim_requested_mode="off",
        memory_reclaim_mode="off",
        rocm_graphs_requested=False,
        rocm_graphs_enabled=False,
        request_timeout_seconds=int(REQUEST_TIMEOUT_SECONDS),
    )
    value["build"] = {
        **mixed.VULKAN_BUILD_SPEC.effective_config(),
        "timeout_seconds": int(BUILD_TIMEOUT_SECONDS),
    }
    value["server"]["max_prefill_tokens_per_cycle"] = 128
    value["model"] = {
        "accelerator_weight_upload_mib_per_second": (
            mixed.ACCELERATOR_WEIGHT_UPLOAD_MIB_PER_SECOND
        ),
        "vulkan_decode_weight_prewarm": mixed.VULKAN_DECODE_WEIGHT_PREWARM,
        "vulkan_decode_weight_prewarm_mib_per_second": (
            mixed.VULKAN_DECODE_WEIGHT_PREWARM_MIB_PER_SECOND
        ),
    }
    value["runtime"].update(
        {
            "prefix_cache_requested_enabled": True,
            "prefix_cache_effective_enabled": False,
            "prefix_cache_effective_reason": "vulkan_correctness_quarantine",
        }
    )
    value["host_safety"] = soak.HOST_THERMAL_POLICY.effective_config()
    value["vulkan_buffer_pool_gb"] = 3.0
    value["workload"] = {
        "cohort_max_tokens": {
            f"cohort_{cohort_index}": {
                f"slot_{slot}": max_tokens
                for slot, max_tokens in enumerate(cohort)
            }
            for cohort_index, cohort in enumerate(COHORT_MAX_TOKENS)
        },
        "cohort_prompt_words": PROMPT_WORDS,
        "expected_completion_tokens": EXPECTED_COMPLETION_TOKENS,
        "expected_finish_reason": "length",
        "expected_request_count": EXPECTED_REQUESTS,
        "host_mem_available_floor_bytes": HOST_MEMORY_FLOOR_BYTES,
        "host_swap_growth_limit_bytes": HOST_SWAP_GROWTH_LIMIT_BYTES,
        "ignore_eos": True,
        "overall_timeout_seconds": int(OVERALL_TIMEOUT_SECONDS),
        "request_timeout_seconds": int(REQUEST_TIMEOUT_SECONDS),
        "response_oracle": "ascending_zero_padded_integers_prefix_v1",
        "same_process_repetitions": len(COHORT_MAX_TOKENS),
        "simultaneous_dispatch": "per_cohort_thread_barrier",
        "varying_completion_lengths": True,
        "warmup_max_tokens": WARMUP_MAX_TOKENS,
    }
    return value


EFFECTIVE_CONFIG = _effective_config()
mixed.VARIANT_CONFIGS[VARIANT_ID] = EFFECTIVE_CONFIG


METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    "batched_state_cache_active_leases_end": ("leases", "exact", True),
    "batched_state_cache_completed_row_eviction_count": ("count", "sum", True),
    "batched_state_cache_entry_present_end": ("entries", "exact", False),
    "batched_state_cache_explicit_invalidation_count": ("count", "sum", True),
    "batched_state_cache_max_active_leases": ("leases", "max", True),
    "batched_state_cache_park_replacement_eviction_count": ("count", "sum", True),
    "batched_state_cache_resident_end": ("count", "exact", False),
    "batched_state_cache_take_miss_while_leased_count": ("count", "sum", True),
    "batching_error_count": ("count", "sum", True),
    "batching_max_observed_active_requests": ("requests", "max", False),
    "batching_max_observed_batch_size": ("rows", "max", False),
    "binary_build_count": ("count", "sum", True),
    "cohort_count": ("count", "sum", False),
    "completion_token_count": ("tokens", "sum", False),
    "device_fault_event_count": ("count", "sum", True),
    "external_yield_sync_call_count": ("count", "sum", False),
    "external_yield_sync_failure_count": ("count", "sum", True),
    "external_yield_sync_max_ms": ("ms", "max", True),
    "external_yield_sync_slow_count": ("count", "sum", True),
    "external_yield_sync_total_ms": ("ms", "sum", True),
    "gpu_memory_baseline_bytes": ("bytes", "exact", True),
    "gpu_memory_end_bytes": ("bytes", "exact", True),
    "gpu_memory_peak_bytes": ("bytes", "max", True),
    "gpu_memory_peak_growth_bytes": ("bytes", "max", True),
    "gpu_memory_sampler_error_count": ("count", "sum", True),
    "graph_activity_count": ("count", "sum", True),
    "host_mem_available_end_bytes": ("bytes", "exact", True),
    "host_mem_available_min_bytes": ("bytes", "min", False),
    "host_mem_available_start_bytes": ("bytes", "exact", True),
    "host_memory_guard_trip_count": ("count", "sum", True),
    "host_swap_growth_bytes": ("bytes", "max", True),
    "host_swap_used_end_bytes": ("bytes", "exact", True),
    "host_swap_used_peak_bytes": ("bytes", "max", True),
    "host_swap_used_start_bytes": ("bytes", "exact", True),
    "host_temperature_end_millicelsius": ("millicelsius", "exact", True),
    "host_temperature_peak_millicelsius": ("millicelsius", "max", True),
    "host_temperature_start_millicelsius": ("millicelsius", "exact", True),
    "host_thermal_cooldown_active_end": ("bool", "exact", True),
    "host_thermal_cooldown_completed_count": ("count", "sum", False),
    "host_thermal_cooldown_peak_millicelsius": ("millicelsius", "max", True),
    "host_thermal_cooldown_sample_count": ("count", "sum", False),
    "host_thermal_cooldown_seconds": ("s", "sum", True),
    "host_thermal_cooldown_stable_sample_count": ("count", "exact", False),
    "host_thermal_cooldown_timeout_count": ("count", "sum", True),
    "host_thermal_guard_error_count": ("count", "sum", True),
    "host_thermal_guard_trip_count": ("count", "sum", True),
    "host_thermal_pacing_active_end": ("bool", "exact", True),
    "host_thermal_pacing_completed_event_count": ("count", "sum", False),
    "host_thermal_pacing_event_count": ("count", "sum", True),
    "host_thermal_pacing_max_seconds": ("s", "max", True),
    "host_thermal_pacing_max_start_millicelsius": (
        "millicelsius",
        "max",
        True,
    ),
    "host_thermal_pacing_seconds": ("s", "sum", True),
    "kv_blocks_used_end": ("blocks", "exact", True),
    "kv_unaccounted_blocks_end": ("blocks", "exact", True),
    "length_terminated_request_count": ("count", "sum", False),
    "policy_attestation_failure_count": ("count", "sum", True),
    "prefix_cache_active_leases_end": ("leases", "exact", True),
    "prefix_cache_pending_release_entries_end": ("entries", "exact", True),
    "prefix_cache_state_bytes_end": ("bytes", "exact", True),
    "request_count": ("count", "sum", False),
    "request_failure_count": ("count", "sum", True),
    "resident_prefill_active_rows_end": ("rows", "exact", True),
    "resident_prefill_attempt_count": ("count", "sum", False),
    "resident_prefill_completed_row_count": ("rows", "sum", False),
    "resident_prefill_enabled": ("bool", "exact", False),
    "resident_prefill_forward_count": ("count", "sum", False),
    "resident_prefill_initial_decline_count": ("count", "sum", True),
    "resident_prefill_max_batch_size": ("rows", "max", False),
    "resident_prefill_metadata_request_count": ("requests", "sum", False),
    "resident_prefill_route_failure_count": ("count", "sum", True),
    "resident_prefill_row_count": ("rows", "sum", False),
    "resident_recurrent_state_allocation_bytes_end": ("bytes", "exact", True),
    "resident_recurrent_state_buffer_bytes_end": ("bytes", "exact", True),
    "resident_recurrent_state_entries_end": ("entries", "exact", True),
    "response_oracle_failure_count": ("count", "sum", True),
    "shutdown_forced_count": ("count", "sum", True),
    "shutdown_nonzero_count": ("count", "sum", True),
    "snapshot_residue_count": ("count", "sum", True),
}


class ResidentPrefillOracleError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class CohortRun:
    index: int
    results: tuple[mixed.StreamResult, ...]
    started: float
    finished: float


@dataclasses.dataclass
class Evidence:
    values: dict[str, float | int] = dataclasses.field(
        default_factory=lambda: {name: 0 for name in METRIC_DEFINITIONS}
    )
    details: dict[str, Any] = dataclasses.field(default_factory=dict)
    cohorts: list[CohortRun] = dataclasses.field(default_factory=list)


def run_cohort(
    port: int,
    *,
    cohort_index: int,
    max_tokens: tuple[int, ...],
    seed: int,
    absolute_deadline: float,
) -> CohortRun:
    dispatch = threading.Barrier(len(max_tokens) + 1)
    abort = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(max_tokens),
        thread_name_prefix=f"vulkan-resident-{cohort_index}",
    )
    futures: list[concurrent.futures.Future[mixed.StreamResult]] = []
    results: tuple[mixed.StreamResult, ...] | None = None
    primary_error: Exception | None = None
    unfinished: set[concurrent.futures.Future[mixed.StreamResult]] = set()
    started = time.monotonic()

    def request(slot: int, completion_limit: int) -> mixed.StreamResult:
        dispatch.wait(
            timeout=mixed.remaining_until(
                absolute_deadline,
                f"resident cohort {cohort_index} dispatch",
                REQUEST_TIMEOUT_SECONDS,
            )
        )
        return mixed.run_stream(
            port,
            name=f"resident-c{cohort_index}-r{slot}",
            marker=mixed.workload_marker(
                seed, f"resident-prefill-c{cohort_index}-r{slot}"
            ),
            prompt_words=PROMPT_WORDS,
            max_tokens=completion_limit,
            seed=seed + cohort_index * 100 + slot,
            absolute_deadline=absolute_deadline,
            abort_event=abort,
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )

    try:
        futures = [
            pool.submit(request, slot, completion_limit)
            for slot, completion_limit in enumerate(max_tokens)
        ]
        dispatch.wait(
            timeout=mixed.remaining_until(
                absolute_deadline,
                f"resident cohort {cohort_index} dispatch",
                REQUEST_TIMEOUT_SECONDS,
            )
        )
        results = tuple(
            future.result(
                timeout=mixed.remaining_until(
                    absolute_deadline,
                    f"resident cohort {cohort_index} completion",
                    REQUEST_TIMEOUT_SECONDS,
                )
            )
            for future in futures
        )
    except Exception as exc:
        primary_error = exc
    finally:
        abort.set()
        for future in futures:
            future.cancel()
        _, unfinished = concurrent.futures.wait(
            futures,
            timeout=max(0.0, min(10.0, absolute_deadline - time.monotonic())),
        )
        pool.shutdown(wait=False, cancel_futures=True)
    if primary_error is not None:
        if unfinished:
            raise ResidentPrefillOracleError(
                f"{type(primary_error).__name__}: {primary_error}; "
                f"{len(unfinished)} request workers survived cohort cleanup"
            ) from primary_error
        raise primary_error
    if unfinished:
        raise ResidentPrefillOracleError(
            f"{len(unfinished)} request workers survived cohort cleanup"
        )
    assert results is not None
    return CohortRun(cohort_index, results, started, time.monotonic())


def semantic_output_sha256(cohorts: list[CohortRun]) -> str:
    records = []
    for cohort in cohorts:
        for result in cohort.results:
            text, text_error = mixed.streamed_plain_text(result)
            records.append(
                {
                    "cohort": cohort.index,
                    "completion_tokens": result.completion_tokens,
                    "name": result.name,
                    "resident_prefill_used": result.resident_prefill_used,
                    "text": text,
                    "text_error": text_error,
                    "token_ids": result.token_ids,
                }
            )
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def metric_records(values: dict[str, float | int]) -> list[dict[str, Any]]:
    if set(values) != set(METRIC_DEFINITIONS):
        missing = sorted(set(METRIC_DEFINITIONS) - set(values))
        extra = sorted(set(values) - set(METRIC_DEFINITIONS))
        raise ResidentPrefillOracleError(
            f"metric set mismatch: missing={missing}, extra={extra}"
        )
    records = []
    for name in sorted(values):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ResidentPrefillOracleError(
                f"metric {name} is not finite nonnegative numeric evidence"
            )
        unit, aggregation, lower_is_better = METRIC_DEFINITIONS[name]
        records.append(
            {
                "name": name,
                "value": value,
                "unit": unit,
                "aggregation": aggregation,
                "lower_is_better": lower_is_better,
            }
        )
    return records


def execution_identity_failures(
    health: dict[str, Any], binary_sha256: str
) -> list[str]:
    identity = health.get("execution_identity")
    if not isinstance(identity, dict):
        return ["health.execution_identity is missing"]
    failures = []
    for field, expected in (
        ("backend", "vulkan"),
        ("device", "vulkan:0"),
        ("executable_sha256", binary_sha256),
    ):
        if identity.get(field) != expected:
            failures.append(
                f"execution identity {field}={identity.get(field)!r}, expected {expected!r}"
            )
    return failures


def result_failures(
    cohorts: list[CohortRun],
) -> list[str]:
    failures = []
    for cohort, expected_limits in zip(cohorts, COHORT_MAX_TOKENS):
        if len(cohort.results) != len(expected_limits):
            failures.append(
                f"cohort {cohort.index} returned {len(cohort.results)} rows, "
                f"expected {len(expected_limits)}"
            )
            continue
        for result, expected_tokens in zip(cohort.results, expected_limits):
            oracle_failure = mixed.deterministic_response_oracle_failure(result)
            if not result.success:
                failures.append(f"{result.name} failed: {result.error!r}")
            if result.finish_reason != "length":
                failures.append(
                    f"{result.name} finish_reason={result.finish_reason!r}, expected length"
                )
            if result.completion_tokens != expected_tokens:
                failures.append(
                    f"{result.name} emitted {result.completion_tokens} tokens, "
                    f"expected {expected_tokens}"
                )
            if oracle_failure is not None:
                failures.append(f"{result.name} response oracle failed: {oracle_failure}")
    return failures


def execute(model_path: Path, seed: int, evidence: Evidence) -> None:
    deadline = time.monotonic() + OVERALL_TIMEOUT_SECONDS
    binary, binary_sha256, build_seconds = mixed.build_binary(
        deadline,
        mixed.VULKAN_BUILD_SPEC,
        build_timeout_seconds=BUILD_TIMEOUT_SECONDS,
    )
    evidence.values["binary_build_count"] = 1
    evidence.details.update(
        {
            "build_seconds": build_seconds,
            "kiln_binary_sha256": binary_sha256,
        }
    )
    mixed.trace(
        "vulkan_resident_prefill_binary_built",
        build_seconds=build_seconds,
        path=str(binary.relative_to(ROOT)),
        sha256=binary_sha256,
    )

    port = mixed.free_loopback_port()
    run_dir = mixed.create_serving_run_dir(VARIANT_ID)
    adapter_dir = run_dir / "adapters"
    snapshot_dir = run_dir / "model-snapshots"
    config_path = run_dir / "kiln.toml"
    process: Any = None
    server_log: mixed.ServerLog | None = None
    gpu_sampler: soak.GpuMemorySampler | None = None
    host_guard: soak.HostMemoryGuard | None = None
    thermal_guard: soak.HostThermalGuard | None = None
    shutdown: mixed.ShutdownOutcome | None = None
    residue: list[str] = []
    failures: list[str] = []
    measurement_started: float | None = None
    try:
        adapter_dir.mkdir(parents=True, exist_ok=False)
        mixed.write_server_config(
            config_path,
            VARIANT_ID,
            model_path,
            port,
            adapter_dir,
            snapshot_dir,
            rocm_graph_mode="disabled",
        )
        evidence.details["generated_config_sha256"] = mixed.sha256_file(config_path)
        process, server_log = mixed.start_server(
            binary, config_path, VARIANT_ID, mixed.VULKAN_BUILD_SPEC
        )
        host_guard = soak.HostMemoryGuard(process, HOST_MEMORY_FLOOR_BYTES)
        thermal_guard = soak.HostThermalGuard(
            process,
            **soak.HOST_THERMAL_POLICY.guard_kwargs(),
        )
        thermal_guard.start()
        host_guard.start()
        startup_health = mixed.wait_ready(port, process, server_log, deadline)
        startup_debug = mixed.json_request(port, "GET", "/v1/debug/model-state")
        startup_failures = [
            *mixed.attest_runtime(
                VARIANT_ID, startup_health, startup_debug, kv_force_blocks=None
            ),
            *execution_identity_failures(startup_health, binary_sha256),
        ]
        evidence.values["policy_attestation_failure_count"] = len(startup_failures)
        if startup_failures:
            raise ResidentPrefillOracleError(
                "startup runtime attestation failed: " + " | ".join(startup_failures)
            )

        warmup = mixed.run_stream(
            port,
            name="resident-warmup",
            marker=mixed.workload_marker(seed, "resident-prefill-warmup"),
            prompt_words=PROMPT_WORDS,
            max_tokens=WARMUP_MAX_TOKENS,
            seed=seed,
            absolute_deadline=deadline,
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        warmup_failure = mixed.deterministic_response_oracle_failure(warmup)
        if (
            not warmup.success
            or warmup.finish_reason != "length"
            or warmup.completion_tokens != WARMUP_MAX_TOKENS
            or warmup_failure is not None
        ):
            raise ResidentPrefillOracleError(
                "warmup failed: "
                f"success={warmup.success}, finish_reason={warmup.finish_reason!r}, "
                f"completion_tokens={warmup.completion_tokens}, "
                f"oracle={warmup_failure!r}, error={warmup.error!r}"
            )
        before_health = soak.wait_drained(port, deadline, "resident oracle warmup")
        before_debug = mixed.json_request(port, "GET", "/v1/debug/model-state")
        batching_before = mixed.batching_snapshot(before_health)
        prefix_before = soak.prefix_cache_snapshot(before_health)
        batched_before = soak.batched_state_cache_snapshot(
            before_debug, soak.VULKAN_RUNTIME
        )
        resident_before = soak.resident_recurrent_state_snapshot(
            before_debug, soak.VULKAN_RUNTIME
        )
        failures.extend(
            soak.disabled_prefix_cache_failures(prefix_before, phase="oracle baseline")
        )
        failures.extend(
            soak.resident_recurrent_state_drain_failures(
                resident_before, "oracle baseline"
            )
        )
        assert batched_before is not None

        gpu_baseline = soak.gpu_memory_bytes(
            port, process.pid, soak.VULKAN_RUNTIME
        )
        gpu_sampler = soak.GpuMemorySampler(port, process.pid, soak.VULKAN_RUNTIME)
        gpu_sampler.start()
        measurement_started = time.monotonic()
        for cohort_index, max_tokens in enumerate(COHORT_MAX_TOKENS):
            cohort = run_cohort(
                port,
                cohort_index=cohort_index,
                max_tokens=max_tokens,
                seed=seed + 10_000,
                absolute_deadline=deadline,
            )
            evidence.cohorts.append(cohort)
            drained_health = soak.wait_drained(
                port, deadline, f"resident oracle cohort {cohort_index}"
            )
            drained_batching = mixed.batching_snapshot(drained_health)
            mixed.trace(
                "vulkan_resident_prefill_cohort_complete",
                cohort=cohort_index,
                duration_seconds=cohort.finished - cohort.started,
                max_resident_prefill_batch_size=drained_batching[
                    "max_resident_prefill_batch_size"
                ],
                resident_prefill_forwards=drained_batching[
                    "total_resident_prefill_forwards"
                ],
                resident_prefill_rows=drained_batching[
                    "total_resident_prefill_rows"
                ],
            )
        gpu_sampler.close()

        after_health = soak.wait_drained(port, deadline, "resident oracle final drain")
        after_debug = mixed.json_request(port, "GET", "/v1/debug/model-state")
        batching_after = mixed.batching_snapshot(after_health)
        prefix_after = soak.prefix_cache_snapshot(after_health)
        batched_after = soak.batched_state_cache_snapshot(
            after_debug, soak.VULKAN_RUNTIME
        )
        resident_after = soak.resident_recurrent_state_snapshot(
            after_debug, soak.VULKAN_RUNTIME
        )
        assert batched_after is not None and resident_after is not None
        final_policy_failures = [
            *mixed.attest_runtime(
                VARIANT_ID, after_health, after_debug, kv_force_blocks=None
            ),
            *execution_identity_failures(after_health, binary_sha256),
            *mixed.attest_runtime_execution(
                VARIANT_ID, before_health, after_health
            ),
        ]
        evidence.values["policy_attestation_failure_count"] += len(
            final_policy_failures
        )
        failures.extend(final_policy_failures)
        failures.extend(result_failures(evidence.cohorts))
        failures.extend(
            soak.disabled_prefix_cache_failures(prefix_after, phase="oracle final")
        )
        failures.extend(
            soak.resident_recurrent_state_drain_failures(
                resident_after, "oracle final"
            )
        )
        if not mixed.batching_engine_drained(
            after_health["decode_runtime"]["batching_engine"]
        ):
            failures.append("batching engine did not drain after resident cohorts")

        results = [
            result for cohort in evidence.cohorts for result in cohort.results
        ]
        resident_values = soak.resident_prefill_metric_values(
            batching_before, batching_after
        )
        failures.extend(
            soak.resident_prefill_contract_failures(
                resident_values, max_configured_rows=4
            )
        )
        batched_values = soak.batched_state_cache_metric_values(
            batched_before, batched_after
        )
        for name in (
            "batched_state_cache_active_leases_end",
            "batched_state_cache_completed_row_eviction_count",
            "batched_state_cache_explicit_invalidation_count",
            "batched_state_cache_park_replacement_eviction_count",
            "batched_state_cache_take_miss_while_leased_count",
        ):
            if batched_values[name] != 0:
                failures.append(f"{name}={batched_values[name]}, expected 0")
        if batched_values["batched_state_cache_max_active_leases"] > 1:
            failures.append("batched recurrent-state cache held concurrent leases")

        oracle_failures = sum(
            mixed.deterministic_response_oracle_failure(result) is not None
            for result in results
        )
        resident_metadata = sum(result.resident_prefill_used is True for result in results)
        if resident_metadata < 2:
            failures.append(
                "fewer than two responses attested resident-prefill execution"
            )
        gpu_end = soak.gpu_memory_bytes(port, process.pid, soak.VULKAN_RUNTIME)
        gpu_peak = max([gpu_baseline, gpu_end, *gpu_sampler.samples])
        events = server_log.events_since(measurement_started)
        categories = [event.category for event in events]
        sync_values = mixed.external_yield_sync_metric_values(
            before_health, after_health
        )
        evidence.values.update(
            {
                **resident_values,
                **soak.resident_recurrent_state_metric_values(resident_after),
                **sync_values,
                "batched_state_cache_active_leases_end": batched_values[
                    "batched_state_cache_active_leases_end"
                ],
                "batched_state_cache_completed_row_eviction_count": batched_values[
                    "batched_state_cache_completed_row_eviction_count"
                ],
                "batched_state_cache_entry_present_end": batched_values[
                    "batched_state_cache_entry_present_end"
                ],
                "batched_state_cache_explicit_invalidation_count": batched_values[
                    "batched_state_cache_explicit_invalidation_count"
                ],
                "batched_state_cache_max_active_leases": batched_values[
                    "batched_state_cache_max_active_leases"
                ],
                "batched_state_cache_park_replacement_eviction_count": batched_values[
                    "batched_state_cache_park_replacement_eviction_count"
                ],
                "batched_state_cache_resident_end": batched_values[
                    "batched_state_cache_resident_end"
                ],
                "batched_state_cache_take_miss_while_leased_count": batched_values[
                    "batched_state_cache_take_miss_while_leased_count"
                ],
                "batching_error_count": mixed.counter_delta(
                    batching_before, batching_after, "total_errors"
                ),
                "batching_max_observed_active_requests": batching_after[
                    "max_observed_active_requests"
                ],
                "batching_max_observed_batch_size": batching_after[
                    "max_observed_batch_size"
                ],
                "cohort_count": len(evidence.cohorts),
                "completion_token_count": sum(
                    result.completion_tokens for result in results
                ),
                "device_fault_event_count": categories.count("device_fault"),
                "gpu_memory_baseline_bytes": gpu_baseline,
                "gpu_memory_end_bytes": gpu_end,
                "gpu_memory_peak_bytes": gpu_peak,
                "gpu_memory_peak_growth_bytes": max(0, gpu_peak - gpu_baseline),
                "gpu_memory_sampler_error_count": len(gpu_sampler.errors),
                "graph_activity_count": sum(
                    category in {"graph_capture", "graph_fallback", "graph_sync"}
                    for category in categories
                ),
                "kv_blocks_used_end": batching_after["blocks_used"],
                "kv_unaccounted_blocks_end": soak.unaccounted_blocks(
                    batching_after, prefix_after
                ),
                "length_terminated_request_count": sum(
                    result.finish_reason == "length" for result in results
                ),
                "prefix_cache_active_leases_end": prefix_after["active_leases"],
                "prefix_cache_pending_release_entries_end": prefix_after[
                    "pending_release_entries"
                ],
                "prefix_cache_state_bytes_end": prefix_after["cached_state_bytes"],
                "request_count": len(results),
                "request_failure_count": sum(not result.success for result in results),
                "resident_prefill_metadata_request_count": resident_metadata,
                "response_oracle_failure_count": oracle_failures,
            }
        )
        evidence.details.update(
            {
                "cohorts": [
                    {
                        "duration_seconds": cohort.finished - cohort.started,
                        "index": cohort.index,
                        "max_tokens": list(COHORT_MAX_TOKENS[cohort.index]),
                        "resident_prefill_used": [
                            result.resident_prefill_used for result in cohort.results
                        ],
                    }
                    for cohort in evidence.cohorts
                ],
                "semantic_output_sha256": semantic_output_sha256(evidence.cohorts),
            }
        )
        for name in (
            "batching_error_count",
            "device_fault_event_count",
            "external_yield_sync_failure_count",
            "external_yield_sync_slow_count",
            "graph_activity_count",
            "gpu_memory_sampler_error_count",
            "kv_blocks_used_end",
            "kv_unaccounted_blocks_end",
            "prefix_cache_active_leases_end",
            "prefix_cache_pending_release_entries_end",
            "prefix_cache_state_bytes_end",
            "request_failure_count",
            "response_oracle_failure_count",
        ):
            if evidence.values[name] != 0:
                failures.append(f"{name}={evidence.values[name]}, expected 0")
        for name, expected in (
            ("cohort_count", len(COHORT_MAX_TOKENS)),
            ("completion_token_count", EXPECTED_COMPLETION_TOKENS),
            ("length_terminated_request_count", EXPECTED_REQUESTS),
            ("request_count", EXPECTED_REQUESTS),
            ("resident_prefill_enabled", 1),
        ):
            if evidence.values[name] != expected:
                failures.append(
                    f"{name}={evidence.values[name]}, expected {expected}"
                )
        if evidence.values["gpu_memory_peak_growth_bytes"] > GPU_PEAK_GROWTH_LIMIT_BYTES:
            failures.append("GPU memory exceeded the active oracle growth limit")
        if evidence.values["external_yield_sync_call_count"] < 1:
            failures.append("oracle exercised no external-yield synchronization boundary")
        if process.poll() is not None:
            failures.append(f"server exited during the oracle ({process.returncode})")
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        if gpu_sampler is not None:
            gpu_sampler.close()
        if host_guard is not None:
            host_guard.close()
            evidence.values.update(host_guard.metric_values())
        if process is not None:
            if thermal_guard is not None:
                thermal_guard.prepare_for_process_exit()
            try:
                shutdown = mixed.terminate_process(process)
            finally:
                if thermal_guard is not None:
                    thermal_guard.close()
        if thermal_guard is not None:
            evidence.values.update(thermal_guard.metric_values())
            evidence.values.update(thermal_guard.pacing_metric_values())
            evidence.values["host_thermal_guard_error_count"] = len(
                thermal_guard.errors
            )
        if server_log is not None:
            server_log.join()
        residue = mixed.snapshot_payload_residue(snapshot_dir)
        if shutdown is not None:
            evidence.values["shutdown_forced_count"] = int(shutdown.forced)
            evidence.values["shutdown_nonzero_count"] = int(shutdown.returncode != 0)
        evidence.values["snapshot_residue_count"] = len(residue)
        shutil.rmtree(run_dir, ignore_errors=True)

    if host_guard is not None:
        if host_guard.trip_reason is not None:
            failures.append(host_guard.trip_reason)
        failures.extend(f"host memory guard: {error}" for error in host_guard.errors)
        if evidence.values["host_swap_growth_bytes"] > HOST_SWAP_GROWTH_LIMIT_BYTES:
            failures.append("host swap growth exceeded the configured limit")
    if thermal_guard is not None:
        if thermal_guard.trip_reason is not None:
            failures.append(thermal_guard.trip_reason)
        failures.extend(f"host thermal guard: {error}" for error in thermal_guard.errors)
        if evidence.values["host_thermal_cooldown_active_end"] != 0:
            failures.append("host thermal cooldown remained active after teardown")
        if evidence.values["host_thermal_cooldown_completed_count"] != 1:
            failures.append("host thermal cooldown did not complete after teardown")
        if evidence.values["host_thermal_cooldown_timeout_count"] != 0:
            failures.append("host thermal cooldown timed out after teardown")
        if evidence.values["host_thermal_pacing_active_end"] != 0:
            failures.append("host thermal pacing remained active after teardown")
        if (
            evidence.values["host_thermal_pacing_completed_event_count"]
            != evidence.values["host_thermal_pacing_event_count"]
        ):
            failures.append("host thermal pacing events did not all complete")
    if shutdown is None:
        failures.append("server shutdown evidence is missing")
    else:
        if shutdown.forced:
            failures.append("server required forced shutdown")
        if shutdown.returncode != 0:
            failures.append(f"server shutdown returned {shutdown.returncode}")
    if residue:
        failures.append("server left private model snapshot payload")
    if failures:
        raise ResidentPrefillOracleError(" | ".join(dict.fromkeys(failures)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    result_path_value = os.environ.get(RESULT_ENV)
    evidence = Evidence()
    status = "failed"
    details: str | None = None
    try:
        if os.environ.get(VARIANT_ENV, "") != VARIANT_ID:
            raise ResidentPrefillOracleError(
                f"{VARIANT_ENV} must be {VARIANT_ID!r}"
            )
        if not result_path_value:
            raise ResidentPrefillOracleError(f"{RESULT_ENV} is required")
        model_path = args.model_path.resolve(strict=True)
        if not model_path.is_dir():
            raise ResidentPrefillOracleError("--model-path must be a directory")
        execute(model_path, args.seed, evidence)
        status = "passed"
        details = json.dumps(
            evidence.details, sort_keys=True, separators=(",", ":")
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        evidence.details["error"] = error
        details = json.dumps(
            evidence.details, sort_keys=True, separators=(",", ":")
        )
        mixed.trace("vulkan_resident_prefill_oracle_error", details=error)
    result = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "status": status,
        "duration_seconds": time.monotonic() - started,
        "effective_config": EFFECTIVE_CONFIG,
        "metrics": metric_records(evidence.values),
        "tolerances": [],
        "details": mixed.bounded_details(details),
    }
    if not result_path_value:
        print(f"{RESULT_ENV} is required", file=sys.stderr)
        return 2
    try:
        mixed.write_result(Path(result_path_value), result)
    except Exception as exc:
        print(f"cannot write qualification result: {exc}", file=sys.stderr)
        return 2
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
