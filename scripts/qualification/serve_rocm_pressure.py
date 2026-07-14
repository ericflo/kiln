#!/usr/bin/env python3
"""Qualify ROCm serving latency under sustained external memory pressure."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import serve_mixed_load as mixed


ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "rocm-memory-pressure"
VARIANT_ID = "automatic-pressure"
RESULT_ENV = mixed.RESULT_ENV
VARIANT_ENV = mixed.VARIANT_ENV
REQUEST_COUNT = 4
REQUEST_MAX_TOKENS = 384
REQUEST_PROMPT_WORDS = 48
REQUEST_TIMEOUT_SECONDS = 180.0
OVERALL_TIMEOUT_SECONDS = 540.0
PRESSURE_READY_TIMEOUT_SECONDS = 120.0
PRESSURE_HOLD_SECONDS = 30.0
MINIMUM_PRESSURE_DURATION_SECONDS = 20.0
PRESSURE_SAMPLE_INTERVAL_SECONDS = 0.5
MINIMUM_PRESSURE_SAMPLES = 40
TARGET_FREE_FRACTION = 0.0975
MINIMUM_FREE_FRACTION = 0.05
MAXIMUM_QUALIFIED_FREE_FRACTION = 0.10
MAXIMUM_PRESSURE_ITL_MS = 1000.0
MINIMUM_PRESSURE_ITL_GAPS = 32
PEER_CHUNK_MIB = 512
PEER_MAX_ALLOCATION_GIB = 110.0
PEER_HOLD_SECONDS = 180.0
PEER_SHUTDOWN_SECONDS = 30.0
MEMORY_RECOVERY_TIMEOUT_SECONDS = 30.0
MEMORY_RECOVERY_FREE_FRACTION = 0.20
PEER_SCRIPT = ROOT / "scripts/qualification/rocm_pressure_peer.py"


WORKLOAD_CONFIG: dict[str, Any] = {
    "request_count": REQUEST_COUNT,
    "request_max_tokens": REQUEST_MAX_TOKENS,
    "request_prompt_words": REQUEST_PROMPT_WORDS,
    "request_timeout_seconds": int(REQUEST_TIMEOUT_SECONDS),
    "overall_timeout_seconds": int(OVERALL_TIMEOUT_SECONDS),
    "pressure_ready_timeout_seconds": int(PRESSURE_READY_TIMEOUT_SECONDS),
    "pressure_hold_seconds": int(PRESSURE_HOLD_SECONDS),
    "minimum_pressure_duration_seconds": int(MINIMUM_PRESSURE_DURATION_SECONDS),
    "pressure_sample_interval_ms": int(PRESSURE_SAMPLE_INTERVAL_SECONDS * 1000),
    "minimum_pressure_samples": MINIMUM_PRESSURE_SAMPLES,
    "target_free_fraction": TARGET_FREE_FRACTION,
    "minimum_free_fraction": MINIMUM_FREE_FRACTION,
    "maximum_qualified_free_fraction": MAXIMUM_QUALIFIED_FREE_FRACTION,
    "maximum_pressure_itl_ms": int(MAXIMUM_PRESSURE_ITL_MS),
    "minimum_pressure_itl_gaps": MINIMUM_PRESSURE_ITL_GAPS,
    "peer_chunk_mib": PEER_CHUNK_MIB,
    "peer_max_allocation_gib": PEER_MAX_ALLOCATION_GIB,
    "peer_hold_seconds": int(PEER_HOLD_SECONDS),
    "memory_recovery_timeout_seconds": int(MEMORY_RECOVERY_TIMEOUT_SECONDS),
    "memory_recovery_free_fraction": MEMORY_RECOVERY_FREE_FRACTION,
}
EFFECTIVE_CONFIG = mixed._variant_config(
    serving_profile="experimental",
    kv_autoscale_requested=False,
    kv_autoscale_enabled=False,
    memory_reclaim_requested_mode="automatic",
    memory_reclaim_mode="automatic",
    rocm_graphs_requested=False,
    rocm_graphs_enabled=False,
)
EFFECTIVE_CONFIG["workload"] = WORKLOAD_CONFIG
mixed.VARIANT_CONFIGS[VARIANT_ID] = EFFECTIVE_CONFIG
mixed.REQUEST_TIMEOUT_SECONDS = REQUEST_TIMEOUT_SECONDS


METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    "automatic_last_duration_ms": ("ms", "max", True),
    "automatic_last_reclaimed_bytes": ("bytes", "exact", True),
    "automatic_last_target_bytes": ("bytes", "exact", False),
    "automatic_reclaim_attempt_count": ("count", "sum", False),
    "automatic_reclaim_successful_count": ("count", "sum", False),
    "automatic_reclaim_suppressed_count": ("count", "sum", False),
    "automatic_reclaim_zero_yield_count": ("count", "sum", False),
    "automatic_reclaimed_bytes": ("bytes", "sum", False),
    "automatic_zero_yield_streak": ("count", "exact", False),
    "backend_quarantine_count": ("count", "sum", True),
    "batching_decode_forward_ms_max": ("ms", "max", True),
    "batching_total_error_count": ("count", "sum", True),
    "completion_token_count": ("tokens", "sum", False),
    "external_yield_sync_call_count": ("count", "sum", False),
    "external_yield_sync_failure_count": ("count", "sum", True),
    "external_yield_sync_max_ms": ("ms", "max", True),
    "external_yield_sync_slow_count": ("count", "sum", True),
    "external_yield_sync_total_ms": ("ms", "sum", True),
    "graph_event_count": ("count", "sum", True),
    "kv_blocks_end": ("blocks", "exact", False),
    "kv_blocks_start": ("blocks", "exact", False),
    "kv_resize_event_count": ("count", "sum", True),
    "length_terminated_request_count": ("count", "sum", False),
    "memory_reclaim_event_count": ("count", "sum", True),
    "peer_allocated_bytes": ("bytes", "exact", False),
    "peer_exit_code": ("code", "exact", True),
    "pressure_below_ten_percent_sample_count": ("count", "sum", False),
    "pressure_duration_ms": ("ms", "exact", False),
    "pressure_free_fraction_max": ("ratio", "max", True),
    "pressure_free_fraction_min": ("ratio", "min", False),
    "pressure_itl_gap_count": ("count", "sum", False),
    "pressure_itl_ms_max": ("ms", "max", True),
    "pressure_itl_ms_p99": ("ms", "p99", True),
    "pressure_itl_over_1000ms_count": ("count", "sum", True),
    "pressure_sample_count": ("count", "sum", False),
    "recovery_duration_ms": ("ms", "exact", True),
    "request_count": ("count", "sum", False),
    "request_failure_count": ("count", "sum", True),
    "requests_active_through_pressure_count": ("count", "exact", False),
}


PROMETHEUS_SELECTORS = frozenset(
    {
        'kiln_gpu_memory_bytes{kind="free"}',
        'kiln_gpu_memory_bytes{kind="total"}',
        'kiln_memory_reclaim_attempts_total{outcome="reclaimed"}',
        'kiln_memory_reclaim_attempts_total{outcome="zero_yield"}',
        "kiln_memory_reclaim_suppressed_total",
        "kiln_memory_reclaimed_bytes_total",
        'kiln_memory_reclaim_last_bytes{kind="target"}',
        'kiln_memory_reclaim_last_bytes{kind="reclaimed"}',
        "kiln_memory_reclaim_last_duration_seconds",
        "kiln_memory_reclaim_retry_after_seconds",
        "kiln_memory_reclaim_zero_yield_streak",
    }
)
PROMETHEUS_LINE_RE = re.compile(
    r"^(?P<selector>[a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^{}]*\})?)\s+"
    r"(?P<value>[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)$"
)
GOVERNOR_FIELDS = (
    "automatic_attempts",
    "automatic_successful_attempts",
    "automatic_zero_yield_attempts",
    "automatic_suppressed_attempts",
    "automatic_reclaimed_bytes",
    "automatic_last_target_bytes",
    "automatic_last_reclaimed_bytes",
    "automatic_last_duration_us",
    "automatic_retry_after_ms",
    "automatic_zero_yield_streak",
)


@dataclass(frozen=True)
class PressureSample:
    observed: float
    free_fraction: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class PeerShutdown:
    returncode: int
    forced: bool
    output: str


def parse_prometheus_values(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_LINE_RE.fullmatch(line)
        if match is None:
            continue
        selector = match.group("selector")
        if selector not in PROMETHEUS_SELECTORS:
            continue
        if selector in values:
            raise mixed.QualificationError(
                f"Prometheus selector {selector!r} appeared more than once"
            )
        value = float(match.group("value"))
        if not math.isfinite(value) or value < 0:
            raise mixed.QualificationError(
                f"Prometheus selector {selector!r} is not finite and nonnegative"
            )
        values[selector] = value
    missing = sorted(PROMETHEUS_SELECTORS - set(values))
    if missing:
        raise mixed.QualificationError(
            "Prometheus pressure evidence is missing selectors: " + ", ".join(missing)
        )
    return values


def memory_free_fraction(metrics: dict[str, float]) -> float:
    total = metrics['kiln_gpu_memory_bytes{kind="total"}']
    free = metrics['kiln_gpu_memory_bytes{kind="free"}']
    if total <= 0 or free > total:
        raise mixed.QualificationError(
            f"invalid Prometheus GPU memory snapshot total={total}, free={free}"
        )
    return free / total


def governor_snapshot(health: dict[str, Any]) -> dict[str, int]:
    runtime = health.get("decode_runtime")
    governor = runtime.get("memory_governor") if isinstance(runtime, dict) else None
    if not isinstance(governor, dict):
        raise mixed.QualificationError("health memory-governor state is missing")
    snapshot: dict[str, int] = {}
    for field in GOVERNOR_FIELDS:
        value = governor.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise mixed.QualificationError(
                f"memory-governor field {field} must be a nonnegative integer, got {value!r}"
            )
        snapshot[field] = value
    if snapshot["automatic_attempts"] != (
        snapshot["automatic_successful_attempts"]
        + snapshot["automatic_zero_yield_attempts"]
    ):
        raise mixed.QualificationError(
            "automatic reclaim attempts do not equal successful plus zero-yield outcomes"
        )
    if snapshot["automatic_last_reclaimed_bytes"] > snapshot["automatic_reclaimed_bytes"]:
        raise mixed.QualificationError(
            "last automatic reclaim yield exceeds the cumulative reclaimed bytes"
        )
    return snapshot


def delta(before: dict[str, int], after: dict[str, int], field: str) -> int:
    if after[field] < before[field]:
        raise mixed.QualificationError(
            f"counter {field} regressed from {before[field]} to {after[field]}"
        )
    return after[field] - before[field]


def prometheus_governor_values(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "automatic_successful_attempts": metrics[
            'kiln_memory_reclaim_attempts_total{outcome="reclaimed"}'
        ],
        "automatic_zero_yield_attempts": metrics[
            'kiln_memory_reclaim_attempts_total{outcome="zero_yield"}'
        ],
        "automatic_suppressed_attempts": metrics[
            "kiln_memory_reclaim_suppressed_total"
        ],
        "automatic_reclaimed_bytes": metrics["kiln_memory_reclaimed_bytes_total"],
        "automatic_last_target_bytes": metrics[
            'kiln_memory_reclaim_last_bytes{kind="target"}'
        ],
        "automatic_last_reclaimed_bytes": metrics[
            'kiln_memory_reclaim_last_bytes{kind="reclaimed"}'
        ],
        "automatic_last_duration_us": metrics[
            "kiln_memory_reclaim_last_duration_seconds"
        ]
        * 1_000_000.0,
        "automatic_retry_after_ms": metrics[
            "kiln_memory_reclaim_retry_after_seconds"
        ]
        * 1000.0,
        "automatic_zero_yield_streak": metrics[
            "kiln_memory_reclaim_zero_yield_streak"
        ],
    }


def prometheus_health_mismatches(
    metrics: dict[str, float], health_snapshot: dict[str, int]
) -> list[str]:
    failures: list[str] = []
    for field, observed in prometheus_governor_values(metrics).items():
        expected = health_snapshot[field]
        if abs(observed - expected) > 0.5:
            failures.append(
                f"Prometheus {field}={observed:g} disagrees with health {expected}"
            )
    return failures


def read_stable_pressure_evidence(
    port: int, absolute_deadline: float
) -> tuple[dict[str, Any], dict[str, float]]:
    deadline = min(time.monotonic() + 10.0, absolute_deadline)
    last_failure = "no snapshot completed"
    while time.monotonic() < deadline:
        before = parse_prometheus_values(mixed.text_request(port, "/metrics"))
        health = mixed.read_stable_health(
            port, absolute_deadline, "sustained-pressure snapshot"
        )
        after = parse_prometheus_values(mixed.text_request(port, "/metrics"))
        before_governor = prometheus_governor_values(before)
        after_governor = prometheus_governor_values(after)
        if before_governor != after_governor:
            last_failure = "automatic reclaim counters changed across the snapshot"
            time.sleep(0.05)
            continue
        mismatches = prometheus_health_mismatches(
            after, governor_snapshot(health)
        )
        if not mismatches:
            return health, after
        last_failure = " | ".join(mismatches)
        time.sleep(0.05)
    raise mixed.QualificationError(
        "could not obtain stable pressure health/Prometheus evidence: " + last_failure
    )


def pressure_itl_gaps(
    results: list[mixed.StreamResult], started: float, ended: float
) -> list[float]:
    gaps: list[float] = []
    for result in results:
        for before, after in zip(
            result.token_ready_times, result.token_ready_times[1:]
        ):
            if before >= started and after <= ended:
                gaps.append((after - before) * 1000.0)
    return gaps


def load_ready_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise mixed.QualificationError(f"cannot read pressure-peer readiness: {exc}") from exc
    expected_keys = {
        "schema_version",
        "pid",
        "allocated_bytes",
        "target_free_fraction",
        "minimum_free_fraction",
        "baseline",
        "ready",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise mixed.QualificationError("pressure-peer readiness has an unexpected shape")
    for name in ("baseline", "ready"):
        snapshot = value[name]
        if not isinstance(snapshot, dict) or set(snapshot) != {
            "total_bytes",
            "used_bytes",
            "free_bytes",
            "free_fraction",
        }:
            raise mixed.QualificationError(
                f"pressure-peer readiness {name} snapshot has an unexpected shape"
            )
        for field in ("total_bytes", "used_bytes", "free_bytes"):
            field_value = snapshot[field]
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or field_value < 0
            ):
                raise mixed.QualificationError(
                    f"pressure-peer readiness {name}.{field} is invalid"
                )
        if snapshot["total_bytes"] <= 0:
            raise mixed.QualificationError(
                f"pressure-peer readiness {name}.total_bytes is not positive"
            )
        fraction = snapshot["free_fraction"]
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0 <= fraction <= 1
        ):
            raise mixed.QualificationError(
                f"pressure-peer readiness {name}.free_fraction is invalid"
            )
        if snapshot["used_bytes"] + snapshot["free_bytes"] != snapshot["total_bytes"]:
            raise mixed.QualificationError(
                f"pressure-peer readiness {name} byte totals are inconsistent"
            )
        expected_fraction = snapshot["free_bytes"] / snapshot["total_bytes"]
        if abs(float(fraction) - expected_fraction) > 1e-12:
            raise mixed.QualificationError(
                f"pressure-peer readiness {name} free fraction is inconsistent"
            )
    allocated = value["allocated_bytes"]
    if isinstance(allocated, bool) or not isinstance(allocated, int) or allocated <= 0:
        raise mixed.QualificationError("pressure peer reported no positive allocation")
    if value["schema_version"] != 1:
        raise mixed.QualificationError("pressure-peer readiness schema version is not 1")
    if isinstance(value["pid"], bool) or not isinstance(value["pid"], int) or value["pid"] <= 0:
        raise mixed.QualificationError("pressure-peer readiness pid is invalid")
    if value["target_free_fraction"] != TARGET_FREE_FRACTION:
        raise mixed.QualificationError("pressure-peer target free fraction drifted")
    if value["minimum_free_fraction"] != MINIMUM_FREE_FRACTION:
        raise mixed.QualificationError("pressure-peer minimum free fraction drifted")
    ready_fraction = float(value["ready"]["free_fraction"])
    if not MINIMUM_FREE_FRACTION <= ready_fraction <= TARGET_FREE_FRACTION:
        raise mixed.QualificationError(
            f"pressure peer declared readiness at free fraction {ready_fraction:.6f}"
        )
    return value


def start_pressure_peer(ready_path: Path) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        str(PEER_SCRIPT),
        "--ready-file",
        str(ready_path),
        "--target-free-fraction",
        str(TARGET_FREE_FRACTION),
        "--minimum-free-fraction",
        str(MINIMUM_FREE_FRACTION),
        "--chunk-mib",
        str(PEER_CHUNK_MIB),
        "--max-allocation-gib",
        str(PEER_MAX_ALLOCATION_GIB),
        "--hold-seconds",
        str(PEER_HOLD_SECONDS),
    ]
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=mixed.sanitized_environment(dict(os.environ)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def terminate_peer(process: subprocess.Popen[str]) -> PeerShutdown:
    forced = False
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        output, _ = process.communicate(timeout=PEER_SHUTDOWN_SECONDS)
    except subprocess.TimeoutExpired:
        forced = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = process.communicate(timeout=10.0)
    return PeerShutdown(process.returncode, forced, output[-8000:])


def wait_for_peer_ready(
    path: Path,
    peer: subprocess.Popen[str],
    requests: list[concurrent.futures.Future[mixed.StreamResult]],
    absolute_deadline: float,
) -> dict[str, Any]:
    deadline = min(
        time.monotonic() + PRESSURE_READY_TIMEOUT_SECONDS, absolute_deadline
    )
    while time.monotonic() < deadline:
        if path.is_file():
            ready = load_ready_file(path)
            if ready["pid"] != peer.pid:
                raise mixed.QualificationError(
                    f"pressure ready pid {ready['pid']} does not match peer {peer.pid}"
                )
            return ready
        if peer.poll() is not None:
            output, _ = peer.communicate(timeout=5.0)
            raise mixed.QualificationError(
                f"pressure peer exited before readiness ({peer.returncode}): {output[-4000:]}"
            )
        completed = sum(future.done() for future in requests)
        if completed:
            raise mixed.QualificationError(
                f"{completed} serving requests completed before pressure became ready"
            )
        time.sleep(0.1)
    raise TimeoutError("pressure peer did not reach its target before the deadline")


def pressure_samples(
    port: int,
    peer: subprocess.Popen[str],
    requests: list[concurrent.futures.Future[mixed.StreamResult]],
    started: float,
    absolute_deadline: float,
) -> tuple[list[PressureSample], int, float]:
    samples: list[PressureSample] = []
    active_through = REQUEST_COUNT
    target_end = started + PRESSURE_HOLD_SECONDS
    while time.monotonic() < target_end:
        mixed.remaining_until(absolute_deadline, "sustained memory pressure")
        if peer.poll() is not None:
            output, _ = peer.communicate(timeout=5.0)
            raise mixed.QualificationError(
                f"pressure peer exited during hold ({peer.returncode}): {output[-4000:]}"
            )
        active_through = min(
            active_through, REQUEST_COUNT - sum(future.done() for future in requests)
        )
        metrics = parse_prometheus_values(mixed.text_request(port, "/metrics"))
        samples.append(
            PressureSample(time.monotonic(), memory_free_fraction(metrics), metrics)
        )
        delay = min(
            PRESSURE_SAMPLE_INTERVAL_SECONDS,
            max(0.0, target_end - time.monotonic()),
        )
        if delay:
            time.sleep(delay)
    active_through = min(
        active_through, REQUEST_COUNT - sum(future.done() for future in requests)
    )
    return samples, active_through, time.monotonic()


def wait_for_memory_recovery(port: int, absolute_deadline: float) -> float:
    started = time.monotonic()
    deadline = min(started + MEMORY_RECOVERY_TIMEOUT_SECONDS, absolute_deadline)
    last_fraction = 0.0
    while time.monotonic() < deadline:
        values = parse_prometheus_values(mixed.text_request(port, "/metrics"))
        last_fraction = memory_free_fraction(values)
        if last_fraction >= MEMORY_RECOVERY_FREE_FRACTION:
            return (time.monotonic() - started) * 1000.0
        time.sleep(0.25)
    raise TimeoutError(
        "GPU memory did not recover to "
        f"{MEMORY_RECOVERY_FREE_FRACTION:.0%} free; last={last_fraction:.3%}"
    )


def metrics_from_values(values: dict[str, float | int]) -> list[dict[str, Any]]:
    if set(values) != set(METRIC_DEFINITIONS):
        missing = sorted(set(METRIC_DEFINITIONS) - set(values))
        extra = sorted(set(values) - set(METRIC_DEFINITIONS))
        raise mixed.QualificationError(
            f"pressure metric set mismatch: missing={missing}, extra={extra}"
        )
    metrics: list[dict[str, Any]] = []
    for name in sorted(values):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise mixed.QualificationError(f"metric {name} is not finite numeric evidence")
        unit, aggregation, lower_is_better = METRIC_DEFINITIONS[name]
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


def zero_metrics() -> list[dict[str, Any]]:
    values = {name: 0 for name in METRIC_DEFINITIONS}
    values["request_failure_count"] = 1
    return metrics_from_values(values)


def execute(model_path: Path, seed: int) -> tuple[list[dict[str, Any]], str | None]:
    absolute_deadline = time.monotonic() + OVERALL_TIMEOUT_SECONDS
    binary, binary_hash, build_seconds = mixed.build_binary(absolute_deadline)
    mixed.trace(
        "binary_built",
        build_seconds=build_seconds,
        path=str(binary.relative_to(ROOT)),
        sha256=binary_hash,
    )
    port = mixed.free_loopback_port()
    run_dir = ROOT / ".qualification/serving-pressure" / str(os.getpid())
    adapter_dir = run_dir / "adapters"
    snapshot_dir = run_dir / "model-snapshots"
    config_path = run_dir / "kiln.toml"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    mixed.write_server_config(
        config_path, VARIANT_ID, model_path, port, adapter_dir, snapshot_dir
    )
    observed_since = time.monotonic()
    server, server_log = mixed.start_server(binary, config_path, VARIANT_ID)
    shutdown: mixed.ShutdownOutcome | None = None
    snapshot_residue: list[str] = []
    result: tuple[list[dict[str, Any]], str | None] | None = None
    peer: subprocess.Popen[str] | None = None
    peer_shutdown: PeerShutdown | None = None
    pool: concurrent.futures.ThreadPoolExecutor | None = None
    futures: list[concurrent.futures.Future[mixed.StreamResult]] = []
    abort_requests = threading.Event()
    try:
        mixed.wait_ready(port, server, server_log, absolute_deadline)
        health_start = mixed.read_stable_health(
            port, absolute_deadline, "pressure startup snapshot"
        )
        debug_start = mixed.json_request(port, "GET", "/v1/debug/model-state")
        startup_failures = mixed.attest_runtime(
            VARIANT_ID, health_start, debug_start
        )
        if startup_failures:
            raise mixed.QualificationError(
                "pressure startup attestation failed: " + " | ".join(startup_failures)
            )
        warmup = mixed.run_stream(
            port,
            name="warmup",
            marker=mixed.workload_marker(seed, "pressure-warmup"),
            prompt_words=16,
            max_tokens=32,
            seed=seed,
            absolute_deadline=absolute_deadline,
        )
        if not warmup.success:
            raise mixed.QualificationError(
                f"pressure warmup failed: {warmup.error or warmup.finish_reason}"
            )
        health_baseline = mixed.read_stable_health(
            port, absolute_deadline, "pressure baseline snapshot"
        )
        baseline_attestation = mixed.attest_runtime(
            VARIANT_ID, health_baseline, debug_start
        )
        if baseline_attestation:
            raise mixed.QualificationError(
                "pressure baseline attestation failed: "
                + " | ".join(baseline_attestation)
            )
        governor_before = governor_snapshot(health_baseline)
        batching_before = mixed.batching_snapshot(health_baseline)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=REQUEST_COUNT)
        first_tokens = [threading.Event() for _ in range(REQUEST_COUNT)]
        futures = [
            pool.submit(
                mixed.run_stream,
                port,
                name=f"pressure-{index:02d}",
                marker=mixed.workload_marker(seed, f"pressure-{index:02d}"),
                prompt_words=REQUEST_PROMPT_WORDS,
                max_tokens=REQUEST_MAX_TOKENS,
                seed=seed + 10 + index,
                first_token_event=first_tokens[index],
                absolute_deadline=absolute_deadline,
                abort_event=abort_requests,
            )
            for index in range(REQUEST_COUNT)
        ]
        for index, first_token in enumerate(first_tokens):
            if not first_token.wait(
                timeout=mixed.remaining_until(
                    absolute_deadline,
                    f"pressure request {index} first token",
                    REQUEST_TIMEOUT_SECONDS,
                )
            ):
                raise TimeoutError(f"pressure request {index} produced no first token")
        ready_path = run_dir / "pressure-ready.json"
        peer = start_pressure_peer(ready_path)
        ready = wait_for_peer_ready(ready_path, peer, futures, absolute_deadline)
        pressure_started = time.monotonic()
        mixed.trace(
            "pressure_ready",
            allocated_bytes=ready["allocated_bytes"],
            free_fraction=ready["ready"]["free_fraction"],
            request_count=REQUEST_COUNT,
        )
        samples, active_through, pressure_ended = pressure_samples(
            port, peer, futures, pressure_started, absolute_deadline
        )
        pressure_health, pressure_metrics = read_stable_pressure_evidence(
            port, absolute_deadline
        )
        governor_pressure = governor_snapshot(pressure_health)
        peer_shutdown = terminate_peer(peer)
        peer = None
        recovery_ms = wait_for_memory_recovery(port, absolute_deadline)
        results = [
            future.result(
                timeout=mixed.remaining_until(
                    absolute_deadline, "pressure serving requests"
                )
            )
            for future in futures
        ]
        pool.shutdown(wait=True, cancel_futures=False)
        pool = None
        health_end = mixed.read_stable_health(
            port, absolute_deadline, "pressure final snapshot"
        )
        debug_end = mixed.json_request(port, "GET", "/v1/debug/model-state")
        final_attestation = mixed.attest_runtime(VARIANT_ID, health_end, debug_end)
        batching_end = mixed.batching_snapshot(health_end)
        sync_values = mixed.external_yield_sync_metric_values(
            health_baseline, pressure_health
        )
        events = server_log.events_since(observed_since)
        pressure_duration_ms = (pressure_ended - pressure_started) * 1000.0
        free_fractions = [sample.free_fraction for sample in samples]
        gaps = pressure_itl_gaps(results, pressure_started, pressure_ended)
        request_failures = sum(not item.success for item in results)
        successful_tokens = sum(
            item.completion_tokens for item in results if item.success
        )
        attempts = delta(
            governor_before, governor_pressure, "automatic_attempts"
        )
        successes = delta(
            governor_before,
            governor_pressure,
            "automatic_successful_attempts",
        )
        zero_yield = delta(
            governor_before,
            governor_pressure,
            "automatic_zero_yield_attempts",
        )
        suppressed = delta(
            governor_before,
            governor_pressure,
            "automatic_suppressed_attempts",
        )
        reclaimed = delta(
            governor_before, governor_pressure, "automatic_reclaimed_bytes"
        )
        values: dict[str, float | int] = {
            "automatic_last_duration_ms": governor_pressure[
                "automatic_last_duration_us"
            ]
            / 1000.0,
            "automatic_last_reclaimed_bytes": governor_pressure[
                "automatic_last_reclaimed_bytes"
            ],
            "automatic_last_target_bytes": governor_pressure[
                "automatic_last_target_bytes"
            ],
            "automatic_reclaim_attempt_count": attempts,
            "automatic_reclaim_successful_count": successes,
            "automatic_reclaim_suppressed_count": suppressed,
            "automatic_reclaim_zero_yield_count": zero_yield,
            "automatic_reclaimed_bytes": reclaimed,
            "automatic_zero_yield_streak": governor_pressure[
                "automatic_zero_yield_streak"
            ],
            "backend_quarantine_count": int(
                bool((health_end.get("backend_runtime") or {}).get("quarantined"))
            ),
            "batching_decode_forward_ms_max": batching_end[
                "max_decode_forward_ms"
            ],
            "batching_total_error_count": mixed.counter_delta(
                batching_before, batching_end, "total_errors"
            ),
            "completion_token_count": successful_tokens,
            **sync_values,
            "graph_event_count": sum(
                event.category in {"graph_capture", "graph_fallback"}
                for event in events
            ),
            "kv_blocks_end": batching_end["blocks_total"],
            "kv_blocks_start": batching_before["blocks_total"],
            "kv_resize_event_count": sum(
                event.category == "kv_resize" for event in events
            ),
            "length_terminated_request_count": sum(
                item.finish_reason == "length" for item in results
            ),
            "memory_reclaim_event_count": sum(
                event.category == "memory_reclaim" for event in events
            ),
            "peer_allocated_bytes": ready["allocated_bytes"],
            "peer_exit_code": peer_shutdown.returncode,
            "pressure_below_ten_percent_sample_count": sum(
                fraction < MAXIMUM_QUALIFIED_FREE_FRACTION
                for fraction in free_fractions
            ),
            "pressure_duration_ms": pressure_duration_ms,
            "pressure_free_fraction_max": max(free_fractions, default=1.0),
            "pressure_free_fraction_min": min(free_fractions, default=0.0),
            "pressure_itl_gap_count": len(gaps),
            "pressure_itl_ms_max": max(gaps, default=0.0),
            "pressure_itl_ms_p99": mixed.percentile_r7(gaps, 0.99),
            "pressure_itl_over_1000ms_count": sum(
                gap >= MAXIMUM_PRESSURE_ITL_MS for gap in gaps
            ),
            "pressure_sample_count": len(samples),
            "recovery_duration_ms": recovery_ms,
            "request_count": len(results),
            "request_failure_count": request_failures,
            "requests_active_through_pressure_count": active_through,
        }
        failures = [*final_attestation]
        failures.extend(
            prometheus_health_mismatches(pressure_metrics, governor_pressure)
        )
        if server.poll() is not None:
            failures.append(f"server exited during pressure qualification ({server.returncode})")
        if peer_shutdown.forced:
            failures.append("pressure peer required forced termination")
        if peer_shutdown.returncode != 0:
            failures.append(
                f"pressure peer returned {peer_shutdown.returncode}, expected 0: "
                + peer_shutdown.output[-1000:]
            )
        if pressure_duration_ms < MINIMUM_PRESSURE_DURATION_SECONDS * 1000.0:
            failures.append(
                f"pressure lasted only {pressure_duration_ms:.3f} ms"
            )
        if len(samples) < MINIMUM_PRESSURE_SAMPLES:
            failures.append(
                f"pressure sampler collected {len(samples)} values, expected at least "
                f"{MINIMUM_PRESSURE_SAMPLES}"
            )
        if free_fractions and max(free_fractions) >= MAXIMUM_QUALIFIED_FREE_FRACTION:
            failures.append(
                f"pressure free fraction reached {max(free_fractions):.6f}, expected < "
                f"{MAXIMUM_QUALIFIED_FREE_FRACTION:.2f}"
            )
        if free_fractions and min(free_fractions) < MINIMUM_FREE_FRACTION:
            failures.append(
                f"pressure free fraction crossed safety floor: {min(free_fractions):.6f}"
            )
        if active_through != REQUEST_COUNT:
            failures.append(
                f"only {active_through}/{REQUEST_COUNT} requests remained active through pressure"
            )
        if len(gaps) < MINIMUM_PRESSURE_ITL_GAPS:
            failures.append(
                f"only {len(gaps)} server-ready ITL gaps fell wholly inside pressure"
            )
        if gaps and max(gaps) >= MAXIMUM_PRESSURE_ITL_MS:
            failures.append(
                f"maximum pressure ITL was {max(gaps):.3f} ms, expected < "
                f"{MAXIMUM_PRESSURE_ITL_MS:.0f} ms"
            )
        if request_failures:
            failures.append(f"{request_failures}/{REQUEST_COUNT} pressure requests failed")
        for item in results:
            if item.finish_reason != "length" or item.completion_tokens != REQUEST_MAX_TOKENS:
                failures.append(
                    f"{item.name} ended with {item.finish_reason!r}/"
                    f"{item.completion_tokens} tokens, expected length/{REQUEST_MAX_TOKENS}"
                )
        if attempts < 1 or zero_yield < 1 or suppressed < 1:
            failures.append(
                "automatic reclaim did not prove attempt, zero-yield, and suppression "
                f"under active pressure ({attempts}/{zero_yield}/{suppressed})"
            )
        if successes != 0 or reclaimed != 0:
            failures.append(
                "automatic reclaim mutated the pool while requests were active: "
                f"successes={successes}, reclaimed_bytes={reclaimed}"
            )
        if governor_pressure["automatic_last_target_bytes"] <= 0:
            failures.append("automatic reclaim recorded no positive pressure target")
        if governor_pressure["automatic_last_reclaimed_bytes"] != 0:
            failures.append("last active-pressure reclaim attempt reported nonzero yield")
        if values["batching_total_error_count"] != 0:
            failures.append("batching engine recorded an error during pressure")
        if values["external_yield_sync_failure_count"] != 0:
            failures.append("external-yield synchronization failed during pressure")
        if values["external_yield_sync_slow_count"] != 0:
            failures.append("external-yield synchronization reached the 100 ms slow threshold")
        if values["kv_resize_event_count"] != 0 or (
            values["kv_blocks_start"] != values["kv_blocks_end"]
        ):
            failures.append("KV cache resized during fixed-capacity pressure qualification")
        if values["graph_event_count"] != 0:
            failures.append("ROCm graph activity occurred despite the graph-off pressure variant")
        if values["memory_reclaim_event_count"] != 0:
            failures.append("a physical pool reclaim completed while requests were active")
        if values["backend_quarantine_count"] != 0:
            failures.append("backend quarantined during pressure qualification")
        for item in results:
            mixed.trace(
                "pressure_request_result",
                completion_tokens=item.completion_tokens,
                e2e_ms=item.e2e_ms,
                error=item.error,
                finish_reason=item.finish_reason,
                name=item.name,
                pressure_itl_gaps=len(
                    pressure_itl_gaps([item], pressure_started, pressure_ended)
                ),
            )
        mixed.trace(
            "pressure_summary",
            active_through=active_through,
            attempts=attempts,
            free_fraction_max=max(free_fractions, default=0.0),
            free_fraction_min=min(free_fractions, default=0.0),
            itl_max_ms=max(gaps, default=0.0),
            itl_p99_ms=mixed.percentile_r7(gaps, 0.99),
            samples=len(samples),
            suppressed=suppressed,
            zero_yield=zero_yield,
        )
        result = metrics_from_values(values), " | ".join(failures) if failures else None
    finally:
        abort_requests.set()
        if peer is not None:
            peer_shutdown = terminate_peer(peer)
        if pool is not None:
            for future in futures:
                future.cancel()
            concurrent.futures.wait(futures, timeout=10.0)
            pool.shutdown(wait=False, cancel_futures=True)
        shutdown = mixed.terminate_process(server)
        server_log.join()
        snapshot_residue = mixed.snapshot_payload_residue(snapshot_dir)
        mixed.trace(
            "server_shutdown",
            elapsed_ms=shutdown.elapsed_ms,
            forced=shutdown.forced,
            returncode=shutdown.returncode,
            snapshot_residue=snapshot_residue,
        )
        shutil.rmtree(run_dir, ignore_errors=True)

    if result is None or shutdown is None:
        raise AssertionError("pressure qualification completed without a result")
    metrics, details = result
    lifecycle_failures: list[str] = []
    if shutdown.forced:
        lifecycle_failures.append("server required forced termination")
    if shutdown.returncode != 0:
        lifecycle_failures.append(
            f"server shutdown returned {shutdown.returncode}, expected 0"
        )
    if snapshot_residue:
        lifecycle_failures.append(
            "server left private model snapshot payload after shutdown: "
            + ", ".join(snapshot_residue)
        )
    if lifecycle_failures:
        details = " | ".join(filter(None, [details, *lifecycle_failures]))
    return metrics, details


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    result_path_value = os.environ.get(RESULT_ENV)
    variant = os.environ.get(VARIANT_ENV, "")
    if not result_path_value:
        print(f"{RESULT_ENV} is required", file=sys.stderr)
        return 2
    status = "failed"
    details: str | None = None
    metrics = zero_metrics()
    try:
        if variant != VARIANT_ID:
            raise mixed.QualificationError(
                f"{VARIANT_ENV} must be {VARIANT_ID!r}, got {variant!r}"
            )
        model_path = args.model_path.resolve(strict=True)
        if not model_path.is_dir():
            raise mixed.QualificationError("--model-path must be a directory")
        metrics, details = execute(model_path, args.seed)
        status = "passed" if details is None else "failed"
    except Exception as exc:
        details = f"{type(exc).__name__}: {exc}"
        mixed.trace("qualification_error", details=details)
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
