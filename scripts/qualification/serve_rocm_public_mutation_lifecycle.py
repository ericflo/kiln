#!/usr/bin/env python3
"""Qualify public ROCm adapter mutation and maintenance-only KV resize."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import http.client
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import serve_mixed_load as mixed


ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "rocm-public-mutation-lifecycle"
VARIANT_ID = "public-mutation-lifecycle"
ADAPTER_VARIANT_ID = "public-mutation-lifecycle-adapter-arm"
MAINTENANCE_VARIANT_ID = "public-mutation-lifecycle-maintenance-arm"
ADAPTER_NAME = "qualification-adapter"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
FORCED_KV_BLOCKS = 1
GRAPH_CACHE_ENTRIES = 8
GRAPH_CACHE_MAX_BYTES = 1 << 30
MAX_TOKENS = 16
PROMPT_WORDS = 64
WARMUP_ATTEMPTS = 8
OVERALL_TIMEOUT_SECONDS = 1800.0
ARM_ORDER = ("adapter", "maintenance_resize")
RESULT_ENV = mixed.RESULT_ENV
VARIANT_ENV = mixed.VARIANT_ENV


class LifecycleError(RuntimeError):
    pass


def _arm_config(
    *,
    serving_profile: str,
    kv_autoscale: bool,
    rocm_graphs: bool,
) -> dict[str, Any]:
    value = mixed._variant_config(
        serving_profile=serving_profile,
        kv_autoscale_requested=kv_autoscale,
        kv_autoscale_enabled=kv_autoscale,
        memory_reclaim_requested_mode="off",
        memory_reclaim_mode="off",
        rocm_graphs_requested=rocm_graphs,
        rocm_graphs_enabled=rocm_graphs,
    )
    value["runtime"]["rocm_graph_mode"] = (
        "lazy_capture_replay" if rocm_graphs else "disabled"
    )
    return value


ADAPTER_CONFIG = _arm_config(
    serving_profile="experimental",
    kv_autoscale=False,
    rocm_graphs=True,
)
MAINTENANCE_CONFIG = _arm_config(
    serving_profile="maintenance",
    kv_autoscale=True,
    rocm_graphs=False,
)
mixed.VARIANT_CONFIGS[ADAPTER_VARIANT_ID] = ADAPTER_CONFIG
mixed.VARIANT_CONFIGS[MAINTENANCE_VARIANT_ID] = MAINTENANCE_CONFIG

EFFECTIVE_CONFIG: dict[str, Any] = {
    "build": dict(ADAPTER_CONFIG["build"]),
    "runtime": {
        "adapter_arm": dict(ADAPTER_CONFIG["runtime"]),
        "maintenance_resize_arm": {
            **MAINTENANCE_CONFIG["runtime"],
            "kv_force_blocks": FORCED_KV_BLOCKS,
        },
    },
    "server": dict(ADAPTER_CONFIG["server"]),
    "workload": {
        "adapter_files": {
            f"file_{index}": name for index, name in enumerate(ADAPTER_FILES)
        },
        "adapter_input_attestation": (
            "source_and_private_copy_sha256_in_result_details"
        ),
        "adapter_name": ADAPTER_NAME,
        "arm_order": {f"arm_{index}": name for index, name in enumerate(ARM_ORDER)},
        "base_output_restore": "exact_canonical_streamed_semantic_deltas",
        "build_reuse": "one_source_bound_binary_for_both_arms",
        "forced_kv_blocks": FORCED_KV_BLOCKS,
        "graph_cache_entries": GRAPH_CACHE_ENTRIES,
        "graph_cache_max_bytes": GRAPH_CACHE_MAX_BYTES,
        "max_tokens": MAX_TOKENS,
        "overall_timeout_seconds": int(OVERALL_TIMEOUT_SECONDS),
        "prompt_words": PROMPT_WORDS,
        "public_endpoints": {
            "endpoint_0": "POST /v1/adapters/load",
            "endpoint_1": "GET /v1/adapters",
            "endpoint_2": "POST /v1/chat/completions",
            "endpoint_3": "POST /v1/adapters/unload",
            "endpoint_4": "GET /v1/config",
        },
        "warmup_attempts": WARMUP_ATTEMPTS,
    },
}


METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    "adapter_load_count": ("count", "sum", False),
    "adapter_load_ms": ("ms", "exact", True),
    "adapter_revision_header_mismatch_count": ("count", "sum", True),
    "adapter_transition_count": ("count", "sum", False),
    "adapter_unload_count": ("count", "sum", False),
    "adapter_unload_ms": ("ms", "exact", True),
    "adapter_weights_bytes": ("bytes", "exact", False),
    "base_output_mismatch_count": ("count", "sum", True),
    "binary_build_count": ("count", "sum", True),
    "device_fault_count": ("count", "sum", True),
    "dirty_shutdown_count": ("count", "sum", True),
    "forced_resize_actual_blocks": ("blocks", "exact", False),
    "forced_resize_count": ("count", "sum", False),
    "forced_resize_duration_ms": ("ms", "exact", True),
    "forced_resize_from_blocks": ("blocks", "exact", False),
    "forced_resize_released_bytes": ("bytes", "exact", False),
    "forced_resize_wait_ms": ("ms", "exact", True),
    "graph_invalidation_eviction_count": ("count", "sum", False),
    "maintenance_inference_rejection_count": ("count", "sum", False),
    "request_failure_count": ("count", "sum", True),
    "snapshot_residue_count": ("count", "sum", True),
}


@dataclasses.dataclass(frozen=True)
class AdapterArm:
    config_sha256: str
    weights_sha256: str
    weights_bytes: int
    content_revision: str
    generated_config_sha256: str
    base_before_sha256: str
    adapter_output_sha256: str
    base_after_sha256: str
    load_ms: float
    unload_ms: float
    graph_invalidation_evictions: int
    transition_count: int
    device_fault_count: int


@dataclasses.dataclass(frozen=True)
class MaintenanceArm:
    generated_config_sha256: str
    from_blocks: int
    actual_blocks: int
    released_bytes: int
    wait_ms: float
    duration_ms: float
    inference_status: int
    inference_error_code: str
    device_fault_count: int


def canonical_semantic_hash(result: mixed.StreamResult) -> str:
    records: list[Any] = []
    for event in result.semantic_deltas:
        choices = event.get("choices")
        if not isinstance(choices, list):
            raise LifecycleError(f"{result.name} emitted a semantic event without choices")
        records.append(
            [
                {"index": choice.get("index"), "delta": choice.get("delta")}
                for choice in choices
                if isinstance(choice, dict)
            ]
        )
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def json_response(
    port: int,
    method: str,
    path: str,
    body: Any | None = None,
    *,
    timeout: float = 10.0,
) -> tuple[int, dict[str, str], Any]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        payload = None if body is None else json.dumps(body, separators=(",", ":"))
        headers = {
            "Accept": "application/json",
            "User-Agent": "kiln-qualification-lifecycle/1",
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        parsed = json.loads(raw) if raw else None
        return response.status, {key.lower(): value for key, value in response.getheaders()}, parsed
    finally:
        connection.close()


def wait_drained(port: int, deadline: float, label: str) -> dict[str, Any]:
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = mixed.json_request(port, "GET", "/health")
        runtime = last.get("decode_runtime")
        batching = runtime.get("batching_engine") if isinstance(runtime, dict) else None
        graph = mixed.graph_snapshot(last)
        if (
            mixed.batching_engine_drained(batching)
            and graph["active_graph_slot_count"] == 0
            and graph["tracked_decode_owner_count"] == 0
        ):
            return last
        time.sleep(0.05)
    raise TimeoutError(f"{label} did not drain: {last!r}")


def wait_maintenance_ready(
    port: int,
    process: subprocess.Popen[str],
    server_log: mixed.ServerLog,
    deadline: float,
) -> dict[str, Any]:
    last_error = "server not queried"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LifecycleError(
                f"maintenance server exited during startup ({process.returncode}): "
                + "\n".join(server_log.tail)
            )
        try:
            health = mixed.json_request(port, "GET", "/health")
            checks = health.get("checks")
            by_name = {
                check.get("name"): check.get("pass")
                for check in checks
                if isinstance(check, dict)
            } if isinstance(checks, list) else {}
            if (
                health.get("status") == "maintenance"
                and by_name.get("inference_admission") is False
                and by_name.get("inference_prewarm_complete") is True
                and all(
                    value is True
                    for name, value in by_name.items()
                    if name != "inference_admission"
                )
            ):
                return health
            last_error = f"health status/checks not ready: {health.get('status')!r} {by_name!r}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.25)
    raise TimeoutError(f"maintenance readiness failed: {last_error}")


def copy_adapter(source: Path, destination_root: Path) -> tuple[str, str, int]:
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise LifecycleError("--adapter-path must be a directory")
    destination = destination_root / ADAPTER_NAME
    destination.mkdir()
    hashes: dict[str, str] = {}
    for name in ADAPTER_FILES:
        source_file = source / name
        if not source_file.is_file() or source_file.is_symlink():
            raise LifecycleError(f"adapter input {source_file} must be a regular file")
        destination_file = destination / name
        shutil.copy2(source_file, destination_file)
        source_hash = mixed.sha256_file(source_file)
        if mixed.sha256_file(destination_file) != source_hash:
            raise LifecycleError(f"private adapter copy changed {name}")
        hashes[name] = source_hash
    weights_bytes = (source / "adapter_model.safetensors").stat().st_size
    if weights_bytes <= 0:
        raise LifecycleError("adapter weights must be nonempty")
    return (
        hashes["adapter_config.json"],
        hashes["adapter_model.safetensors"],
        weights_bytes,
    )


def assert_stream(result: mixed.StreamResult, label: str) -> None:
    if not result.success:
        raise LifecycleError(f"{label} failed: {result.error or result.finish_reason}")
    if result.finish_reason != "length" or result.completion_tokens != MAX_TOKENS:
        raise LifecycleError(f"{label} did not preserve the fixed output denominator")


def adapter_state(
    port: int, expected_name: str | None, expected_revision: str | None
) -> None:
    health = mixed.json_request(port, "GET", "/health")
    debug = mixed.json_request(port, "GET", "/v1/debug/model-state")
    listed = mixed.json_request(port, "GET", "/v1/adapters")
    expected_count = int(expected_name is not None)
    if health.get("loaded_adapter") != expected_name:
        raise LifecycleError("health loaded adapter does not match the public transition")
    if health.get("loaded_adapter_revision") != expected_revision:
        raise LifecycleError("health loaded adapter revision does not match")
    if health.get("loaded_adapter_count") != expected_count:
        raise LifecycleError("health loaded adapter count does not match")
    adapters = debug.get("adapters")
    if not isinstance(adapters, dict):
        raise LifecycleError("debug adapter state is missing")
    if adapters.get("loaded_adapter") != expected_name:
        raise LifecycleError("debug loaded adapter does not match")
    if adapters.get("loaded_adapter_revision") != expected_revision:
        raise LifecycleError("debug loaded adapter revision does not match")
    identity = listed.get("loaded_adapter_identity")
    if expected_name is None:
        if identity is not None or listed.get("loaded_adapter") is not None:
            raise LifecycleError("adapter list retained a loaded identity after unload")
    elif not isinstance(identity, dict) or identity != {
        "name": expected_name,
        "content_revision": expected_revision,
    }:
        raise LifecycleError("adapter list loaded identity does not match")


def warm_graph(port: int, seed: int, deadline: float) -> None:
    for index in range(WARMUP_ATTEMPTS):
        result = mixed.run_stream(
            port,
            name=f"lifecycle-warmup-{index}",
            marker=mixed.workload_marker(seed, f"lifecycle-warmup-{index}"),
            prompt_words=PROMPT_WORDS,
            max_tokens=MAX_TOKENS,
            seed=seed + index,
            absolute_deadline=deadline,
        )
        assert_stream(result, f"graph warmup {index}")
        graph = mixed.graph_snapshot(wait_drained(port, deadline, "graph warmup"))
        if graph["capture_successes"] > 0 and graph["replay_successes"] > 0:
            return
    raise LifecycleError("adapter arm established no ROCm graph capture/replay outcome")


def run_adapter_arm(
    binary: Path,
    model_path: Path,
    adapter_path: Path,
    seed: int,
    deadline: float,
) -> AdapterArm:
    started = time.monotonic()
    port = mixed.free_loopback_port()
    run_dir = ROOT / ".qualification/serving" / f"public-lifecycle-adapter-{os.getpid()}"
    adapter_dir = run_dir / "adapters"
    snapshot_dir = run_dir / "model-snapshots"
    config_path = run_dir / "kiln.toml"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    config_hash, weights_hash, weights_bytes = copy_adapter(adapter_path, adapter_dir)
    mixed.write_server_config(
        config_path,
        ADAPTER_VARIANT_ID,
        model_path,
        port,
        adapter_dir,
        snapshot_dir,
        rocm_graph_mode="lazy_capture_replay",
        rocm_graph_cache_entries=GRAPH_CACHE_ENTRIES,
        rocm_graph_cache_max_bytes=GRAPH_CACHE_MAX_BYTES,
    )
    generated_config_hash = mixed.sha256_file(config_path)
    process, server_log = mixed.start_server(
        binary, config_path, ADAPTER_VARIANT_ID
    )
    residue: list[str] = []
    try:
        health = mixed.wait_ready(port, process, server_log, deadline)
        debug = mixed.json_request(port, "GET", "/v1/debug/model-state")
        failures = mixed.attest_runtime(
            ADAPTER_VARIANT_ID,
            health,
            debug,
            rocm_graph_cache_entries=GRAPH_CACHE_ENTRIES,
            rocm_graph_cache_max_bytes=GRAPH_CACHE_MAX_BYTES,
        )
        if failures:
            raise LifecycleError("; ".join(failures))
        adapter_state(port, None, None)
        warm_graph(port, seed, deadline)

        base_before = mixed.run_stream(
            port,
            name="lifecycle-base-before",
            marker=mixed.workload_marker(seed, "lifecycle-base-identity"),
            prompt_words=PROMPT_WORDS,
            max_tokens=MAX_TOKENS,
            seed=seed + 100,
            absolute_deadline=deadline,
        )
        assert_stream(base_before, "base-before request")
        if (base_before.loaded_adapter, base_before.loaded_adapter_revision) != (
            "base",
            "base",
        ):
            raise LifecycleError("base-before response omitted the authoritative base headers")
        graph_before = mixed.graph_snapshot(wait_drained(port, deadline, "base-before"))

        load_started = time.monotonic()
        status, _, load = json_response(
            port,
            "POST",
            "/v1/adapters/load",
            {"name": ADAPTER_NAME, "allow_quarantined": False},
            timeout=mixed.remaining_until(deadline, "adapter load", 120.0),
        )
        load_ms = (time.monotonic() - load_started) * 1000.0
        if status != 200 or not isinstance(load, dict):
            raise LifecycleError(f"adapter load returned HTTP {status}: {load!r}")
        content_revision = load.get("content_revision")
        if (
            load.get("status") != "loaded"
            or load.get("name") != ADAPTER_NAME
            or not isinstance(content_revision, str)
            or len(content_revision) != 64
        ):
            raise LifecycleError(f"adapter load response is incomplete: {load!r}")
        int(content_revision, 16)
        adapter_state(port, ADAPTER_NAME, content_revision)

        adapter_output = mixed.run_stream(
            port,
            name="lifecycle-adapter-output",
            marker=mixed.workload_marker(seed, "lifecycle-base-identity"),
            prompt_words=PROMPT_WORDS,
            max_tokens=MAX_TOKENS,
            seed=seed + 100,
            absolute_deadline=deadline,
        )
        assert_stream(adapter_output, "adapter request")
        if (
            adapter_output.loaded_adapter,
            adapter_output.loaded_adapter_revision,
        ) != (ADAPTER_NAME, content_revision):
            raise LifecycleError("adapter response headers do not bind the loaded revision")
        wait_drained(port, deadline, "adapter inference")

        unload_started = time.monotonic()
        status, _, unload = json_response(
            port,
            "POST",
            "/v1/adapters/unload",
            {},
            timeout=mixed.remaining_until(deadline, "adapter unload", 120.0),
        )
        unload_ms = (time.monotonic() - unload_started) * 1000.0
        if status != 200 or unload != {"status": "unloaded"}:
            raise LifecycleError(f"adapter unload returned HTTP {status}: {unload!r}")
        adapter_state(port, None, None)

        base_after = mixed.run_stream(
            port,
            name="lifecycle-base-after",
            marker=mixed.workload_marker(seed, "lifecycle-base-identity"),
            prompt_words=PROMPT_WORDS,
            max_tokens=MAX_TOKENS,
            seed=seed + 100,
            absolute_deadline=deadline,
        )
        assert_stream(base_after, "base-after request")
        if (base_after.loaded_adapter, base_after.loaded_adapter_revision) != (
            "base",
            "base",
        ):
            raise LifecycleError("base-after response retained stale adapter headers")
        before_hash = canonical_semantic_hash(base_before)
        after_hash = canonical_semantic_hash(base_after)
        if before_hash != after_hash:
            raise LifecycleError("base output changed after adapter load/unload")

        health_end = wait_drained(port, deadline, "adapter final")
        debug_end = mixed.json_request(port, "GET", "/v1/debug/model-state")
        failures = mixed.attest_runtime(
            ADAPTER_VARIANT_ID,
            health_end,
            debug_end,
            rocm_graph_cache_entries=GRAPH_CACHE_ENTRIES,
            rocm_graph_cache_max_bytes=GRAPH_CACHE_MAX_BYTES,
        )
        if failures:
            raise LifecycleError("; ".join(failures))
        graph_end = mixed.graph_snapshot(health_end)
        invalidations = (
            graph_end["invalidation_evictions"]
            - graph_before["invalidation_evictions"]
        )
        if invalidations < 1:
            raise LifecycleError("adapter transitions did not invalidate a captured graph")
        events = server_log.events_since(started)
        transitions = [event for event in events if event.category == "adapter_transition"]
        reasons = [event.fields.get("reason") for event in transitions]
        if reasons != ["adapter_load_endpoint", "adapter_unload_endpoint"]:
            raise LifecycleError(f"adapter transition log reasons drifted: {reasons!r}")
        device_faults = sum(event.category == "device_fault" for event in events)
        if device_faults:
            raise LifecycleError(f"adapter arm observed {device_faults} device faults")
        if process.poll() is not None:
            raise LifecycleError("adapter server exited before teardown")
        return AdapterArm(
            config_sha256=config_hash,
            weights_sha256=weights_hash,
            weights_bytes=weights_bytes,
            content_revision=content_revision,
            generated_config_sha256=generated_config_hash,
            base_before_sha256=before_hash,
            adapter_output_sha256=canonical_semantic_hash(adapter_output),
            base_after_sha256=after_hash,
            load_ms=load_ms,
            unload_ms=unload_ms,
            graph_invalidation_evictions=invalidations,
            transition_count=len(transitions),
            device_fault_count=device_faults,
        )
    finally:
        shutdown = mixed.terminate_process(process)
        server_log.join()
        residue = mixed.snapshot_payload_residue(snapshot_dir)
        shutil.rmtree(run_dir, ignore_errors=True)
        if shutdown.forced or shutdown.returncode != 0 or residue:
            raise LifecycleError(
                "adapter teardown failed: "
                f"forced={shutdown.forced}, returncode={shutdown.returncode}, residue={residue}"
            )


def run_maintenance_arm(
    binary: Path,
    model_path: Path,
    seed: int,
    deadline: float,
) -> MaintenanceArm:
    del seed
    started = time.monotonic()
    port = mixed.free_loopback_port()
    run_dir = ROOT / ".qualification/serving" / f"public-lifecycle-maintenance-{os.getpid()}"
    adapter_dir = run_dir / "adapters"
    snapshot_dir = run_dir / "model-snapshots"
    config_path = run_dir / "kiln.toml"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    mixed.write_server_config(
        config_path,
        MAINTENANCE_VARIANT_ID,
        model_path,
        port,
        adapter_dir,
        snapshot_dir,
        rocm_graph_mode="disabled",
        rocm_graph_cache_entries=GRAPH_CACHE_ENTRIES,
        rocm_graph_cache_max_bytes=GRAPH_CACHE_MAX_BYTES,
        kv_force_blocks=FORCED_KV_BLOCKS,
    )
    generated_config_hash = mixed.sha256_file(config_path)
    process, server_log = mixed.start_server(
        binary, config_path, MAINTENANCE_VARIANT_ID
    )
    try:
        health = wait_maintenance_ready(port, process, server_log, deadline)
        resize_event: mixed.ObservedEvent | None = None
        while time.monotonic() < deadline:
            candidates = [
                event
                for event in server_log.events_since(started)
                if event.category == "kv_resize"
                and event.fields.get("reason") == "forced_configuration"
            ]
            if candidates:
                resize_event = candidates[0]
                break
            time.sleep(0.05)
        if resize_event is None:
            raise LifecycleError("maintenance arm observed no forced-configuration resize")
        fields = resize_event.fields
        if fields.get("outcome") != "completed":
            raise LifecycleError(f"forced resize did not complete: {fields!r}")
        integer_fields: dict[str, int] = {}
        for name in (
            "from_blocks",
            "requested_blocks",
            "actual_blocks",
            "released_bytes",
        ):
            value = fields.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LifecycleError(f"forced resize {name} is invalid: {value!r}")
            integer_fields[name] = value
        if integer_fields["from_blocks"] <= FORCED_KV_BLOCKS:
            raise LifecycleError("forced-resize fixture did not start above its target")
        if integer_fields["requested_blocks"] != FORCED_KV_BLOCKS:
            raise LifecycleError("forced resize requested the wrong target")
        if integer_fields["actual_blocks"] != FORCED_KV_BLOCKS:
            raise LifecycleError("forced resize did not reach the exact shrink target")
        numeric_fields: dict[str, float] = {}
        for name in ("wait_ms", "duration_ms"):
            value = fields.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise LifecycleError(f"forced resize {name} is invalid: {value!r}")
            numeric_fields[name] = float(value)
            if not math.isfinite(numeric_fields[name]) or numeric_fields[name] < 0:
                raise LifecycleError(f"forced resize {name} is not finite/nonnegative")

        config = mixed.json_request(port, "GET", "/v1/config")
        kv_cache = config.get("kv_cache")
        if not isinstance(kv_cache, dict) or kv_cache.get("num_blocks") != FORCED_KV_BLOCKS:
            raise LifecycleError("/v1/config did not observe the exact forced KV target")
        debug = mixed.json_request(port, "GET", "/v1/debug/model-state")
        failures = mixed.attest_runtime(
            MAINTENANCE_VARIANT_ID,
            health,
            debug,
            rocm_graph_cache_entries=GRAPH_CACHE_ENTRIES,
            rocm_graph_cache_max_bytes=GRAPH_CACHE_MAX_BYTES,
            kv_force_blocks=FORCED_KV_BLOCKS,
        )
        if failures:
            raise LifecycleError("; ".join(failures))
        batching_before_rejection = mixed.batching_snapshot(health)
        status, _, body = json_response(
            port,
            "POST",
            "/v1/chat/completions",
            mixed.request_body("maintenance admission probe", 1, 0),
        )
        error = body.get("error") if isinstance(body, dict) else None
        error_code = error.get("code") if isinstance(error, dict) else None
        if status != 503 or error_code != "inference_disabled_by_profile":
            raise LifecycleError(
                f"maintenance inference admission returned HTTP {status}: {body!r}"
            )
        health_end = mixed.json_request(port, "GET", "/health")
        batching_after_rejection = mixed.batching_snapshot(health_end)
        for field in (
            "total_admission_calls",
            "total_prefill_forwards",
            "total_decode_forwards",
            "blocks_used",
        ):
            if batching_after_rejection[field] != batching_before_rejection[field]:
                raise LifecycleError(
                    f"rejected maintenance request changed batching {field}"
                )
        events = server_log.events_since(started)
        forced_events = [
            event
            for event in events
            if event.category == "kv_resize"
            and event.fields.get("reason") == "forced_configuration"
        ]
        if len(forced_events) != 1:
            raise LifecycleError(f"expected one forced resize, got {len(forced_events)}")
        device_faults = sum(event.category == "device_fault" for event in events)
        if device_faults:
            raise LifecycleError(f"maintenance arm observed {device_faults} device faults")
        if process.poll() is not None:
            raise LifecycleError("maintenance server exited before teardown")
        return MaintenanceArm(
            generated_config_sha256=generated_config_hash,
            from_blocks=integer_fields["from_blocks"],
            actual_blocks=integer_fields["actual_blocks"],
            released_bytes=integer_fields["released_bytes"],
            wait_ms=numeric_fields["wait_ms"],
            duration_ms=numeric_fields["duration_ms"],
            inference_status=status,
            inference_error_code=error_code,
            device_fault_count=device_faults,
        )
    finally:
        shutdown = mixed.terminate_process(process)
        server_log.join()
        residue = mixed.snapshot_payload_residue(snapshot_dir)
        shutil.rmtree(run_dir, ignore_errors=True)
        if shutdown.forced or shutdown.returncode != 0 or residue:
            raise LifecycleError(
                "maintenance teardown failed: "
                f"forced={shutdown.forced}, returncode={shutdown.returncode}, residue={residue}"
            )


def metric_records(values: dict[str, float | int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in sorted(METRIC_DEFINITIONS):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LifecycleError(f"metric {name} is not numeric: {value!r}")
        if not math.isfinite(float(value)):
            raise LifecycleError(f"metric {name} is not finite: {value!r}")
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


def zero_metrics() -> list[dict[str, Any]]:
    values = {name: 0 for name in METRIC_DEFINITIONS}
    values["binary_build_count"] = 1
    values["dirty_shutdown_count"] = 1
    values["request_failure_count"] = 1
    return metric_records(values)


def execute(
    model_path: Path, adapter_path: Path, seed: int
) -> tuple[list[dict[str, Any]], str]:
    deadline = time.monotonic() + OVERALL_TIMEOUT_SECONDS
    binary, binary_hash, build_seconds = mixed.build_binary(deadline)
    mixed.trace(
        "public_mutation_lifecycle_binary_built",
        build_seconds=build_seconds,
        path=str(binary.relative_to(ROOT)),
        sha256=binary_hash,
    )
    adapter = run_adapter_arm(binary, model_path, adapter_path, seed, deadline)
    maintenance = run_maintenance_arm(binary, model_path, seed, deadline)
    values: dict[str, float | int] = {
        "adapter_load_count": 1,
        "adapter_load_ms": adapter.load_ms,
        "adapter_revision_header_mismatch_count": 0,
        "adapter_transition_count": adapter.transition_count,
        "adapter_unload_count": 1,
        "adapter_unload_ms": adapter.unload_ms,
        "adapter_weights_bytes": adapter.weights_bytes,
        "base_output_mismatch_count": 0,
        "binary_build_count": 1,
        "device_fault_count": adapter.device_fault_count + maintenance.device_fault_count,
        "dirty_shutdown_count": 0,
        "forced_resize_actual_blocks": maintenance.actual_blocks,
        "forced_resize_count": 1,
        "forced_resize_duration_ms": maintenance.duration_ms,
        "forced_resize_from_blocks": maintenance.from_blocks,
        "forced_resize_released_bytes": maintenance.released_bytes,
        "forced_resize_wait_ms": maintenance.wait_ms,
        "graph_invalidation_eviction_count": adapter.graph_invalidation_evictions,
        "maintenance_inference_rejection_count": 1,
        "request_failure_count": 0,
        "snapshot_residue_count": 0,
    }
    details = json.dumps(
        {
            "adapter_config": adapter.config_sha256,
            "adapter_weights": adapter.weights_sha256,
            "adapter_content_revision": adapter.content_revision,
            "adapter_arm_config": adapter.generated_config_sha256,
            "adapter_output": adapter.adapter_output_sha256,
            "base_after": adapter.base_after_sha256,
            "base_before": adapter.base_before_sha256,
            "kiln_binary": binary_hash,
            "maintenance_arm_config": maintenance.generated_config_sha256,
            "maintenance_inference_error_code": maintenance.inference_error_code,
            "maintenance_inference_status": maintenance.inference_status,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return metric_records(values), details


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    result_path_value = os.environ.get(RESULT_ENV)
    variant = os.environ.get(VARIANT_ENV, "")
    if not result_path_value:
        print(f"{RESULT_ENV} is required", file=os.sys.stderr)
        return 2
    status = "failed"
    details: str | None = None
    metrics = zero_metrics()
    try:
        if variant != VARIANT_ID:
            raise LifecycleError(
                f"{VARIANT_ENV} must be {VARIANT_ID!r}, got {variant!r}"
            )
        model_path = args.model_path.resolve(strict=True)
        adapter_path = args.adapter_path.resolve(strict=True)
        if not model_path.is_dir():
            raise LifecycleError("--model-path must be a directory")
        metrics, details = execute(model_path, adapter_path, args.seed)
        status = "passed"
    except Exception as exc:
        details = f"{type(exc).__name__}: {exc}"
        mixed.trace("public_mutation_lifecycle_error", details=details)
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
        print(f"cannot write qualification result: {exc}", file=os.sys.stderr)
        return 2
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
