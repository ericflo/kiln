#!/usr/bin/env python3
"""Run a source-bound Vulkan serving baseline with concurrency scaling evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import serve_mixed_load as mixed


ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "vulkan-serving-baseline"
VARIANT_ID = "vulkan-serving-baseline"
RESULT_ENV = mixed.RESULT_ENV
VARIANT_ENV = mixed.VARIANT_ENV
BUILD_TIMEOUT_SECONDS = 900.0
OVERALL_TIMEOUT_SECONDS = 1800.0
REQUEST_TIMEOUT_SECONDS = 180.0
WARMUP_MAX_TOKENS = 16
MAX_TOKENS = 32
PAUSE_GATE_MS = 2_000.0
STALL_EVIDENCE_MS = 250.0
WAVES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("single", (16,)),
    ("batch-4", (16, 64, 192, 384)),
    ("batch-8", (16, 32, 64, 96, 192, 384, 512, 768)),
    (
        "saturation-12",
        (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024),
    ),
)
EXPECTED_REQUESTS = sum(len(prompt_words) for _, prompt_words in WAVES)
EXPECTED_COMPLETION_TOKENS = EXPECTED_REQUESTS * MAX_TOKENS


def _effective_config() -> dict[str, Any]:
    value = mixed._variant_config(
        serving_profile="experimental",
        kv_autoscale_requested=False,
        kv_autoscale_enabled=False,
        memory_reclaim_requested_mode="off",
        memory_reclaim_mode="off",
        rocm_graphs_requested=False,
        rocm_graphs_enabled=False,
    )
    value["build"] = {
        **mixed.VULKAN_BUILD_SPEC.effective_config(),
        "timeout_seconds": int(BUILD_TIMEOUT_SECONDS),
    }
    value["workload"] = {
        "expected_completion_tokens": EXPECTED_COMPLETION_TOKENS,
        "expected_finish_reason": "length",
        "expected_request_count": EXPECTED_REQUESTS,
        "ignore_eos": True,
        "max_tokens": MAX_TOKENS,
        "overall_timeout_seconds": int(OVERALL_TIMEOUT_SECONDS),
        "pause_gate_ms": int(PAUSE_GATE_MS),
        "request_timeout_seconds": int(REQUEST_TIMEOUT_SECONDS),
        "semantic_output_evidence": "ordered_canonical_sha256",
        "simultaneous_dispatch": "per_wave_thread_barrier",
        "stall_evidence_ms": int(STALL_EVIDENCE_MS),
        "warmup_max_tokens": WARMUP_MAX_TOKENS,
        "waves": {
            name: {
                f"slot_{index}": words
                for index, words in enumerate(prompt_words)
            }
            for name, prompt_words in WAVES
        },
    }
    return value


EFFECTIVE_CONFIG = _effective_config()
mixed.VARIANT_CONFIGS[VARIANT_ID] = EFFECTIVE_CONFIG


def _wave_metrics(prefix: str) -> dict[str, tuple[str, str, bool]]:
    return {
        f"{prefix}_completion_token_count": ("tokens", "sum", False),
        f"{prefix}_duration_seconds": ("s", "exact", True),
        f"{prefix}_e2e_ms_p99": ("ms", "p99", True),
        f"{prefix}_itl_ms_p99": ("ms", "p99", True),
        f"{prefix}_output_tokens_per_second": ("tokens/s", "rate", False),
        f"{prefix}_request_count": ("count", "sum", False),
        f"{prefix}_request_failure_count": ("count", "sum", True),
        f"{prefix}_ttft_ms_p99": ("ms", "p99", True),
    }


METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    **{
        name: definition
        for wave_name, _ in WAVES
        for name, definition in _wave_metrics(wave_name.replace("-", "_")).items()
    },
    "attributed_itl_outlier_count": ("count", "sum", True),
    "batching_batched_decode_forward_count": ("count", "sum", False),
    "batching_decode_forward_count": ("count", "sum", False),
    "batching_decode_row_count": ("rows", "sum", False),
    "batching_max_observed_active_requests": ("requests", "max", False),
    "batching_max_observed_batch_size": ("rows", "max", False),
    "batching_prefill_forward_count": ("count", "sum", False),
    "batching_prefill_layer_count": ("layers", "sum", False),
    "batching_prefill_layer_yield_count": ("count", "sum", False),
    "batching_total_error_count": ("count", "sum", True),
    "binary_build_count": ("count", "sum", True),
    "completion_token_count": ("tokens", "sum", False),
    "device_fault_count": ("count", "sum", True),
    "external_yield_sync_call_count": ("count", "sum", False),
    "external_yield_sync_failure_count": ("count", "sum", True),
    "external_yield_sync_max_ms": ("ms", "max", True),
    "external_yield_sync_slow_count": ("count", "sum", True),
    "external_yield_sync_total_ms": ("ms", "sum", True),
    "graph_activity_count": ("count", "sum", True),
    "itl_pause_count": ("count", "sum", True),
    "itl_stall_count": ("count", "sum", True),
    "kv_blocks_end": ("blocks", "exact", False),
    "kv_blocks_start": ("blocks", "exact", False),
    "kv_resize_event_count": ("count", "sum", True),
    "length_terminated_request_count": ("count", "sum", False),
    "memory_reclaim_event_count": ("count", "sum", True),
    "memory_sample_count": ("count", "sum", False),
    "memory_sampler_error_count": ("count", "sum", True),
    "output_token_throughput_per_second": ("tokens/s", "rate", False),
    "peak_gpu_memory_used_bytes": ("bytes", "max", True),
    "policy_attestation_failure_count": ("count", "sum", True),
    "prompt_token_count": ("tokens", "sum", False),
    "request_count": ("count", "sum", False),
    "request_failure_count": ("count", "sum", True),
    "semantic_output_record_count": ("count", "sum", False),
    "shutdown_forced_count": ("count", "sum", True),
    "shutdown_nonzero_count": ("count", "sum", True),
    "snapshot_residue_count": ("count", "sum", True),
    "unexplained_itl_outlier_count": ("count", "sum", True),
}


class VulkanBaselineError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class WaveRun:
    name: str
    results: tuple[mixed.StreamResult, ...]
    started: float
    finished: float


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise VulkanBaselineError(f"{label} is not a canonical SHA-256 identity")
    return value


def canonical_semantic_sha256(waves: tuple[WaveRun, ...]) -> str:
    records: list[Any] = []
    for wave in waves:
        for result in wave.results:
            semantic = []
            for event in result.semantic_deltas:
                choices = event.get("choices")
                if not isinstance(choices, list):
                    raise VulkanBaselineError(
                        f"{result.name} emitted a semantic event without choices"
                    )
                semantic.append(
                    [
                        {"delta": choice.get("delta"), "index": choice.get("index")}
                        for choice in choices
                        if isinstance(choice, dict)
                    ]
                )
            records.append(
                {
                    "completion_tokens": result.completion_tokens,
                    "name": result.name,
                    "prompt_tokens": result.prompt_tokens,
                    "semantic": semantic,
                }
            )
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def execution_identity_failures(
    health: dict[str, Any], debug: dict[str, Any], binary_sha256: str
) -> list[str]:
    failures: list[str] = []
    identity = health.get("execution_identity")
    provenance = (debug.get("model") or {}).get("execution_provenance")
    if not isinstance(identity, dict):
        return ["health.execution_identity is missing"]
    if not isinstance(provenance, dict):
        return ["debug model execution provenance is missing"]

    expected_summary = {
        "provenance_type": provenance.get("provenance_type"),
        "provenance_sha256": provenance.get("provenance_sha256"),
        "backend": (provenance.get("backend") or {}).get("name"),
        "device": (provenance.get("backend") or {}).get("device"),
        "executable_sha256": (provenance.get("build") or {}).get(
            "executable_sha256"
        ),
        "numerical_runtime_sha256": (provenance.get("backend") or {}).get(
            "numerical_runtime_sha256"
        ),
        "kernel_contract_sha256": (provenance.get("kernels") or {}).get(
            "contract_sha256"
        ),
        "inference_dtype": (provenance.get("precision") or {}).get(
            "inference_dtype"
        ),
        "training_policy": (provenance.get("precision") or {}).get(
            "training_policy"
        ),
        "effective_server_config_sha256": (
            provenance.get("configuration") or {}
        ).get("effective_server_config_sha256"),
        "effective_environment_sha256": (
            provenance.get("configuration") or {}
        ).get("effective_environment_sha256"),
    }
    for field, expected in expected_summary.items():
        if identity.get(field) != expected:
            failures.append(
                f"health execution identity {field} disagrees with debug provenance"
            )
    if identity.get("provenance_type") != "kiln.execution-provenance.v1":
        failures.append("execution provenance type is not v1")
    if identity.get("backend") != "vulkan":
        failures.append(
            f"execution backend={identity.get('backend')!r}, expected 'vulkan'"
        )
    if identity.get("device") != "vulkan:0":
        failures.append(
            f"execution device={identity.get('device')!r}, expected 'vulkan:0'"
        )
    if identity.get("executable_sha256") != binary_sha256:
        failures.append("execution provenance does not bind the source-built binary")
    for field in (
        "provenance_sha256",
        "executable_sha256",
        "numerical_runtime_sha256",
        "kernel_contract_sha256",
        "effective_server_config_sha256",
        "effective_environment_sha256",
    ):
        try:
            require_sha256(identity.get(field), f"execution identity {field}")
        except VulkanBaselineError as exc:
            failures.append(str(exc))
    features = (provenance.get("kernels") or {}).get("compiled_features")
    if not isinstance(features, list) or "vulkan" not in features:
        failures.append("execution kernel contract does not include the Vulkan feature")
    if isinstance(features, list) and "rocm" in features:
        failures.append("Vulkan execution kernel contract unexpectedly includes ROCm")
    source_dirty = (provenance.get("build") or {}).get("source_dirty")
    if source_dirty is not False:
        failures.append(
            f"execution provenance source_dirty={source_dirty!r}, expected false"
        )
    return failures


def run_wave(
    port: int,
    wave_index: int,
    name: str,
    prompt_words: tuple[int, ...],
    seed: int,
    absolute_deadline: float,
) -> WaveRun:
    dispatch = threading.Barrier(len(prompt_words) + 1)

    def request(slot: int, words: int) -> mixed.StreamResult:
        dispatch.wait(
            timeout=mixed.remaining_until(
                absolute_deadline, f"{name} dispatch", REQUEST_TIMEOUT_SECONDS
            )
        )
        return mixed.run_stream(
            port,
            name=f"{name}-{slot:02d}",
            marker=mixed.workload_marker(seed, f"vulkan-{wave_index}-{slot}"),
            prompt_words=words,
            max_tokens=MAX_TOKENS,
            seed=seed + wave_index * 100 + slot,
            absolute_deadline=absolute_deadline,
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(prompt_words), thread_name_prefix=f"vulkan-{name}"
    ) as pool:
        futures = [
            pool.submit(request, slot, words)
            for slot, words in enumerate(prompt_words)
        ]
        dispatch.wait(
            timeout=mixed.remaining_until(
                absolute_deadline, f"{name} dispatch", REQUEST_TIMEOUT_SECONDS
            )
        )
        results = tuple(
            future.result(
                timeout=mixed.remaining_until(
                    absolute_deadline, f"{name} completion", REQUEST_TIMEOUT_SECONDS
                )
            )
            for future in futures
        )
    return WaveRun(name, results, started, time.monotonic())


def _delta(before: dict[str, float | int], after: dict[str, float | int], field: str) -> int:
    value = mixed.counter_delta(before, after, field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VulkanBaselineError(f"batching delta {field} is not an integer")
    return value


def metric_values(
    *,
    waves: tuple[WaveRun, ...],
    warmup: mixed.StreamResult,
    before: dict[str, Any],
    after: dict[str, Any],
    events: list[mixed.ObservedEvent],
    memory_samples: list[int],
    memory_errors: list[str],
    policy_failures: list[str],
) -> dict[str, float | int]:
    results = [result for wave in waves for result in wave.results]
    successes = [result for result in results if result.success]
    completion_tokens = sum(result.completion_tokens for result in successes)
    prompt_tokens = sum(result.prompt_tokens for result in successes)
    started = min((result.started for result in results), default=time.monotonic())
    finished = max((result.finished for result in results), default=started)
    duration = max(finished - started, 1e-9)
    itls = [gap for result in successes for gap in result.itl_ms]
    attributed, unexplained = mixed.classify_itl_outliers(
        warmup.itl_ms, successes, events
    )
    batching_before = mixed.batching_snapshot(before)
    batching_after = mixed.batching_snapshot(after)
    external_sync = mixed.external_yield_sync_metric_values(before, after)
    categories = [event.category for event in events]
    graph_categories = {"graph_capture", "graph_fallback", "graph_sync"}
    values: dict[str, float | int] = {
        "attributed_itl_outlier_count": attributed,
        "batching_batched_decode_forward_count": _delta(
            batching_before, batching_after, "total_batched_decode_forwards"
        ),
        "batching_decode_forward_count": _delta(
            batching_before, batching_after, "total_decode_forwards"
        ),
        "batching_decode_row_count": _delta(
            batching_before, batching_after, "total_decode_rows"
        ),
        "batching_max_observed_active_requests": max(
            int(batching_before["max_observed_active_requests"]),
            int(batching_after["max_observed_active_requests"]),
        ),
        "batching_max_observed_batch_size": max(
            int(batching_before["max_observed_batch_size"]),
            int(batching_after["max_observed_batch_size"]),
        ),
        "batching_prefill_forward_count": _delta(
            batching_before, batching_after, "total_prefill_forwards"
        ),
        "batching_prefill_layer_count": _delta(
            batching_before, batching_after, "total_prefill_layers"
        ),
        "batching_prefill_layer_yield_count": _delta(
            batching_before, batching_after, "total_prefill_layer_yields"
        ),
        "batching_total_error_count": _delta(
            batching_before, batching_after, "total_errors"
        ),
        "binary_build_count": 1,
        "completion_token_count": completion_tokens,
        "device_fault_count": categories.count("device_fault"),
        "graph_activity_count": sum(
            category in graph_categories for category in categories
        ),
        "itl_pause_count": sum(gap > PAUSE_GATE_MS for gap in itls),
        "itl_stall_count": sum(gap > STALL_EVIDENCE_MS for gap in itls),
        "kv_blocks_end": int(batching_after["blocks_total"]),
        "kv_blocks_start": int(batching_before["blocks_total"]),
        "kv_resize_event_count": categories.count("kv_resize"),
        "length_terminated_request_count": sum(
            result.finish_reason == "length" for result in successes
        ),
        "memory_reclaim_event_count": categories.count("memory_reclaim"),
        "memory_sample_count": len(memory_samples),
        "memory_sampler_error_count": len(memory_errors),
        "output_token_throughput_per_second": completion_tokens / duration,
        "peak_gpu_memory_used_bytes": max(memory_samples, default=0),
        "policy_attestation_failure_count": len(policy_failures),
        "prompt_token_count": prompt_tokens,
        "request_count": len(results),
        "request_failure_count": len(results) - len(successes),
        "semantic_output_record_count": len(results),
        "shutdown_forced_count": 0,
        "shutdown_nonzero_count": 0,
        "snapshot_residue_count": 0,
        "unexplained_itl_outlier_count": unexplained,
        **external_sync,
    }
    for wave in waves:
        prefix = wave.name.replace("-", "_")
        wave_successes = [result for result in wave.results if result.success]
        wave_duration = max(wave.finished - wave.started, 1e-9)
        wave_tokens = sum(result.completion_tokens for result in wave_successes)
        values.update(
            {
                f"{prefix}_completion_token_count": wave_tokens,
                f"{prefix}_duration_seconds": wave_duration,
                f"{prefix}_e2e_ms_p99": mixed.percentile_r7(
                    [result.e2e_ms for result in wave_successes], 0.99
                ),
                f"{prefix}_itl_ms_p99": mixed.percentile_r7(
                    [gap for result in wave_successes for gap in result.itl_ms],
                    0.99,
                ),
                f"{prefix}_output_tokens_per_second": wave_tokens / wave_duration,
                f"{prefix}_request_count": len(wave.results),
                f"{prefix}_request_failure_count": len(wave.results)
                - len(wave_successes),
                f"{prefix}_ttft_ms_p99": mixed.percentile_r7(
                    [result.ttft_ms for result in wave_successes], 0.99
                ),
            }
        )
    return values


def metric_records(values: dict[str, float | int]) -> list[dict[str, Any]]:
    if set(values) != set(METRIC_DEFINITIONS):
        missing = sorted(set(METRIC_DEFINITIONS) - set(values))
        extra = sorted(set(values) - set(METRIC_DEFINITIONS))
        raise VulkanBaselineError(
            f"metric set mismatch: missing={missing}, extra={extra}"
        )
    records: list[dict[str, Any]] = []
    for name in sorted(values):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise VulkanBaselineError(f"metric {name} is not finite numeric evidence")
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


def qualification_failures(
    values: dict[str, float | int], waves: tuple[WaveRun, ...]
) -> list[str]:
    failures: list[str] = []
    exact = {
        "request_count": EXPECTED_REQUESTS,
        "request_failure_count": 0,
        "length_terminated_request_count": EXPECTED_REQUESTS,
        "completion_token_count": EXPECTED_COMPLETION_TOKENS,
        "semantic_output_record_count": EXPECTED_REQUESTS,
        "policy_attestation_failure_count": 0,
        "device_fault_count": 0,
        "kv_resize_event_count": 0,
        "memory_reclaim_event_count": 0,
        "graph_activity_count": 0,
        "batching_total_error_count": 0,
        "external_yield_sync_failure_count": 0,
        "external_yield_sync_slow_count": 0,
        "itl_pause_count": 0,
        "unexplained_itl_outlier_count": 0,
        "memory_sampler_error_count": 0,
    }
    for metric, expected in exact.items():
        if values.get(metric) != expected:
            failures.append(
                f"{metric}={values.get(metric)!r}, expected exact value {expected}"
            )
    if values["kv_blocks_start"] != values["kv_blocks_end"]:
        failures.append("KV block capacity changed while dynamic resizing was disabled")
    if values["batching_batched_decode_forward_count"] < 1:
        failures.append("concurrency sweep executed no batched decode forward")
    if values["batching_decode_row_count"] <= values["batching_decode_forward_count"]:
        failures.append("decode counters do not prove multi-row batching")
    if values["batching_max_observed_batch_size"] < 2:
        failures.append("concurrency sweep never observed a multi-row decode batch")
    if values["batching_prefill_forward_count"] < EXPECTED_REQUESTS:
        failures.append("not every measured request reached a prefill forward")
    if values["external_yield_sync_call_count"] < 1:
        failures.append("Vulkan load exercised no external-yield synchronization boundary")
    if values["memory_sample_count"] < 1:
        failures.append("GPU memory sampler collected no values")
    if values["peak_gpu_memory_used_bytes"] <= 0:
        failures.append("GPU memory sampler never observed positive usage")

    for wave, (_, expected_words) in zip(waves, WAVES):
        if len(wave.results) != len(expected_words):
            failures.append(
                f"{wave.name} returned {len(wave.results)} requests, expected {len(expected_words)}"
            )
        for result in wave.results:
            if result.loaded_adapter is not None or result.loaded_adapter_revision is not None:
                failures.append(f"{result.name} unexpectedly reported an adapter identity")
            if result.completion_tokens != MAX_TOKENS:
                failures.append(
                    f"{result.name} emitted {result.completion_tokens} tokens, "
                    f"expected {MAX_TOKENS}"
                )
            if result.actor_queue_ms is None:
                failures.append(f"{result.name} omitted terminal actor timing metadata")
    return failures


def execute(model_path: Path, seed: int) -> tuple[list[dict[str, Any]], str]:
    deadline = time.monotonic() + OVERALL_TIMEOUT_SECONDS
    binary, binary_sha256, build_seconds = mixed.build_binary(
        deadline,
        mixed.VULKAN_BUILD_SPEC,
        build_timeout_seconds=BUILD_TIMEOUT_SECONDS,
    )
    mixed.trace(
        "vulkan_binary_built",
        build_seconds=build_seconds,
        path=str(binary.relative_to(ROOT)),
        sha256=binary_sha256,
    )
    port = mixed.free_loopback_port()
    run_dir = ROOT / ".qualification/serving" / f"{VARIANT_ID}-{os.getpid()}"
    adapter_dir = run_dir / "adapters"
    snapshot_dir = run_dir / "model-snapshots"
    config_path = run_dir / "kiln.toml"
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
    config_sha256 = mixed.sha256_file(config_path)
    process = subprocess.Popen(
        [str(binary), "--config", str(config_path), "serve"],
        cwd=ROOT,
        env=mixed.server_environment(VARIANT_ID, mixed.VULKAN_BUILD_SPEC),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    server_log = mixed.ServerLog(process.stdout)
    server_log.start()
    sampler = mixed.MemorySampler(port)
    values: dict[str, float | int] | None = None
    evidence: dict[str, Any] | None = None
    failures: list[str] = []
    shutdown: mixed.ShutdownOutcome | None = None
    residue: list[str] = []
    try:
        startup_health = mixed.wait_ready(port, process, server_log, deadline)
        startup_debug = mixed.json_request(port, "GET", "/v1/debug/model-state")
        policy_failures = [
            *mixed.attest_runtime(
                VARIANT_ID, startup_health, startup_debug, kv_force_blocks=None
            ),
            *execution_identity_failures(
                startup_health, startup_debug, binary_sha256
            ),
        ]
        failures.extend(policy_failures)
        if policy_failures:
            raise VulkanBaselineError(
                "startup runtime attestation failed: " + " | ".join(policy_failures)
            )
        warmup = mixed.run_stream(
            port,
            name="warmup",
            marker=mixed.workload_marker(seed, "vulkan-warmup"),
            prompt_words=16,
            max_tokens=WARMUP_MAX_TOKENS,
            seed=seed,
            absolute_deadline=deadline,
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        if not warmup.success or warmup.finish_reason != "length":
            raise VulkanBaselineError(
                f"Vulkan warmup failed: {warmup.error or warmup.finish_reason}"
            )
        before = mixed.json_request(port, "GET", "/health")
        measurement_started = time.monotonic()
        sampler.start()
        waves = tuple(
            run_wave(port, index, name, words, seed + 1_000, deadline)
            for index, (name, words) in enumerate(WAVES)
        )
        sampler.close()
        after = mixed.json_request(port, "GET", "/health")
        final_debug = mixed.json_request(port, "GET", "/v1/debug/model-state")
        policy_failures = [
            *mixed.attest_runtime(
                VARIANT_ID, after, final_debug, kv_force_blocks=None
            ),
            *execution_identity_failures(after, final_debug, binary_sha256),
        ]
        failures.extend(policy_failures)
        if not mixed.batching_engine_drained(
            (after.get("decode_runtime") or {}).get("batching_engine")
        ):
            failures.append("batching engine did not drain after the concurrency sweep")
        events = server_log.events_since(measurement_started)
        values = metric_values(
            waves=waves,
            warmup=warmup,
            before=before,
            after=after,
            events=events,
            memory_samples=list(sampler.samples),
            memory_errors=list(sampler.errors),
            policy_failures=policy_failures,
        )
        failures.extend(qualification_failures(values, waves))
        if process.poll() is not None:
            failures.append(
                f"Vulkan server exited during measured load ({process.returncode})"
            )
        evidence = {
            "effective_environment_sha256": after["execution_identity"][
                "effective_environment_sha256"
            ],
            "effective_server_config_sha256": after["execution_identity"][
                "effective_server_config_sha256"
            ],
            "execution_provenance_sha256": after["execution_identity"][
                "provenance_sha256"
            ],
            "generated_config_sha256": config_sha256,
            "kernel_contract_sha256": after["execution_identity"][
                "kernel_contract_sha256"
            ],
            "kiln_binary_sha256": binary_sha256,
            "semantic_output_sha256": canonical_semantic_sha256(waves),
        }
    finally:
        sampler.close()
        shutdown = mixed.terminate_process(process)
        server_log.join()
        residue = mixed.snapshot_payload_residue(snapshot_dir)
        shutil.rmtree(run_dir, ignore_errors=True)

    if values is None or evidence is None or shutdown is None:
        raise VulkanBaselineError("Vulkan baseline ended without complete evidence")
    values["shutdown_forced_count"] = int(shutdown.forced)
    values["shutdown_nonzero_count"] = int(shutdown.returncode != 0)
    values["snapshot_residue_count"] = len(residue)
    if shutdown.forced:
        failures.append("server exceeded the graceful shutdown deadline")
    if shutdown.returncode != 0:
        failures.append(
            f"server shutdown returned {shutdown.returncode}, expected zero"
        )
    if residue:
        failures.append("server left private model snapshot payload after shutdown")
    if failures:
        raise VulkanBaselineError(" | ".join(dict.fromkeys(failures)))
    return metric_records(values), json.dumps(
        evidence, sort_keys=True, separators=(",", ":")
    )


def zero_metrics() -> list[dict[str, Any]]:
    return metric_records({name: 0 for name in METRIC_DEFINITIONS})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    variant = os.environ.get(VARIANT_ENV, "")
    result_path_value = os.environ.get(RESULT_ENV)
    if not result_path_value:
        print(f"{RESULT_ENV} is required", file=sys.stderr)
        return 2
    status = "failed"
    details: str | None = None
    metrics = zero_metrics()
    try:
        if variant != VARIANT_ID:
            raise VulkanBaselineError(
                f"{VARIANT_ENV} must be {VARIANT_ID!r}, got {variant!r}"
            )
        model_path = args.model_path.resolve(strict=True)
        if not model_path.is_dir():
            raise VulkanBaselineError("--model-path must be a directory")
        metrics, details = execute(model_path, args.seed)
        status = "passed"
    except Exception as exc:
        details = f"{type(exc).__name__}: {exc}"
        mixed.trace("vulkan_baseline_error", details=details)
    result = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "status": status,
        "duration_seconds": time.monotonic() - started,
        "effective_config": EFFECTIVE_CONFIG,
        "metrics": metrics,
        "tolerances": [],
        "details": mixed.bounded_details(details),
    }
    try:
        mixed.write_result(Path(result_path_value), result)
    except Exception as exc:
        print(f"cannot write qualification result: {exc}", file=sys.stderr)
        return 2
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
