#!/usr/bin/env python3
"""Run a continuous-process ROCm mixed-load soak and emit bounded evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
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
RESULT_ENV = "KILN_QUALIFICATION_CASE_RESULT"
VARIANT_ENV = "KILN_QUALIFICATION_VARIANT_ID"
CASE_ID = "continuous-mixed-load"
RUNTIME_VARIANT = "autoscale-off"
DEFAULT_MEMORY_GROWTH_LIMIT_BYTES = 512 * 1024 * 1024
WAVE_CONCURRENCY = (1, 8, 12, 8)
PROMPT_WORDS = (16, 32, 64, 128, 256, 384, 512, 768, 96, 192, 1024, 48)
MAX_TOKENS = 32
CANCEL_EVERY_WAVES = 5
CANCELLATION_MAX_TOKENS = 512
CANCELLATION_PROMPT_WORDS = 48
QUALIFICATION_DURATION_SECONDS = 1800.0
MAX_STEADY_STATE_WARMUP_WAVES = 16
GRAPH_CACHE_MAX = 12
MIN_STABILIZATION_CYCLES = 4
MAX_STABILIZATION_CYCLES = 8
REQUIRED_STABLE_CYCLES = 2
STABILIZATION_GPU_DELTA_LIMIT_BYTES = 64 * 1024 * 1024
STABILIZATION_RSS_DELTA_LIMIT_BYTES = 16 * 1024 * 1024
SETUP_DEADLINE_SECONDS = 840.0

METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    "attributed_itl_outlier_count": ("count", "sum", True),
    "batching_error_count": ("count", "sum", True),
    "batching_max_observed_active_requests": ("requests", "max", False),
    "batching_max_observed_batch_size": ("rows", "max", False),
    "cancellation_confirmed_count": ("count", "sum", False),
    "completion_token_count": ("tokens", "sum", False),
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
    "rss_baseline_bytes": ("bytes", "exact", True),
    "rss_end_bytes": ("bytes", "exact", True),
    "rss_growth_bytes": ("bytes", "exact", True),
    "rss_peak_bytes": ("bytes", "max", True),
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
    "stabilization_request_count": ("count", "sum", False),
    "stabilization_stable_cycle_count": ("count", "sum", False),
    "ttft_ms_p50": ("ms", "p50", True),
    "ttft_ms_p99": ("ms", "p99", True),
    "ttft_ms_p999": ("ms", "p99.9", True),
    "unexplained_itl_outlier_count": ("count", "sum", True),
    "wave_count": ("count", "sum", False),
    "zero_token_response_count": ("count", "sum", True),
}


class SoakError(RuntimeError):
    pass


def effective_config(
    minimum_duration_seconds: float, memory_growth_limit_bytes: int
) -> dict[str, Any]:
    base = mixed.VARIANT_CONFIGS[RUNTIME_VARIANT]
    return {
        "build": base["build"],
        "runtime": base["runtime"],
        "server": base["server"],
        "soak": {
            "cancellation_after_semantic_deltas": mixed.CANCELLATION_AFTER_DELTAS,
            "cancellation_max_tokens": CANCELLATION_MAX_TOKENS,
            "cancellation_prompt_words": CANCELLATION_PROMPT_WORDS,
            "cancel_every_waves": CANCEL_EVERY_WAVES,
            "max_tokens": MAX_TOKENS,
            "rocm_graph_cache_max": GRAPH_CACHE_MAX,
            "memory_growth_limit_bytes": memory_growth_limit_bytes,
            "minimum_duration_seconds": minimum_duration_seconds,
            "outlier_absolute_ms": int(mixed.OUTLIER_ABSOLUTE_MS),
            "outlier_history_size": mixed.OUTLIER_HISTORY_SIZE,
            "outlier_multiplier": int(mixed.OUTLIER_MULTIPLIER),
            "prompt_identity": "fixed_by_slot_measured_unique_by_epoch_warmup",
            "prompt_words": {
                f"slot_{index}": words for index, words in enumerate(PROMPT_WORDS)
            },
            "request_ignore_eos": True,
            "setup_deadline_seconds": int(SETUP_DEADLINE_SECONDS),
            "stabilization_gpu_delta_limit_bytes": (
                STABILIZATION_GPU_DELTA_LIMIT_BYTES
            ),
            "stabilization_max_cycles": MAX_STABILIZATION_CYCLES,
            "stabilization_min_cycles": MIN_STABILIZATION_CYCLES,
            "stabilization_required_stable_cycles": REQUIRED_STABLE_CYCLES,
            "stabilization_rss_delta_limit_bytes": (
                STABILIZATION_RSS_DELTA_LIMIT_BYTES
            ),
            "steady_state_warmup_max_waves": MAX_STEADY_STATE_WARMUP_WAVES,
            "wave_concurrency": {
                f"wave_{index}": concurrency
                for index, concurrency in enumerate(WAVE_CONCURRENCY)
            },
        },
    }


def rss_bytes(pid: int) -> int:
    status = Path(f"/proc/{pid}/status")
    for line in status.read_text(encoding="utf-8").splitlines():
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) != 3 or fields[2] != "kB":
            break
        value = int(fields[1]) * 1024
        if value >= 0:
            return value
    raise SoakError(f"cannot read a valid VmRSS value for server pid {pid}")


def gpu_memory_bytes(port: int) -> int:
    value = mixed.parse_prometheus_used_bytes(mixed.text_request(port, "/metrics"))
    if value is None:
        raise SoakError("server metrics omitted kiln_gpu_memory_bytes{kind=\"used\"}")
    return value


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


def run_wave(
    port: int,
    *,
    wave: int,
    base_seed: int,
    deadline: float,
    phase: str = "measured",
    prompt_epoch: int | None = None,
) -> list[mixed.StreamResult]:
    concurrency = WAVE_CONCURRENCY[wave % len(WAVE_CONCURRENCY)]
    abort = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    futures: list[concurrent.futures.Future[mixed.StreamResult]] = []
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
                    prompt_words=PROMPT_WORDS[slot],
                    max_tokens=MAX_TOKENS,
                    seed=base_seed + wave * 100 + slot,
                    absolute_deadline=deadline,
                    abort_event=abort,
                )
            )
        return [
            future.result(timeout=mixed.remaining_until(deadline, "soak wave"))
            for future in futures
        ]
    finally:
        abort.set()
        for future in futures:
            future.cancel()
        _, unfinished = concurrent.futures.wait(
            futures,
            timeout=max(0.0, min(10.0, deadline - time.monotonic())),
        )
        pool.shutdown(wait=False, cancel_futures=True)
        if unfinished:
            raise SoakError(f"{len(unfinished)} request workers survived wave cleanup")


def run_cancellation(
    port: int,
    *,
    wave: int,
    base_seed: int,
    phase: str,
    deadline: float,
) -> str | None:
    role = f"soak-{phase}-cancel-w{wave:05d}"
    wave_seed = base_seed + wave * 100
    cancelled = mixed.run_stream(
        port,
        name=role,
        marker=mixed.workload_marker(wave_seed, role),
        prompt_words=CANCELLATION_PROMPT_WORDS,
        max_tokens=CANCELLATION_MAX_TOKENS,
        seed=wave_seed + 99,
        cancel_after=mixed.CANCELLATION_AFTER_DELTAS,
        absolute_deadline=deadline,
    )
    confirmed, _ = mixed.wait_for_cancellation_and_drain(
        port, cancelled.marker, deadline
    )
    if (
        not cancelled.cancelled
        or len(cancelled.semantic_times) < mixed.CANCELLATION_AFTER_DELTAS
        or not confirmed
    ):
        return f"cancellation was not confirmed in {phase} wave {wave}"
    return None


def metrics_from_values(values: dict[str, float | int]) -> list[dict[str, Any]]:
    if set(values) != set(METRIC_DEFINITIONS):
        missing = sorted(set(METRIC_DEFINITIONS) - set(values))
        extra = sorted(set(values) - set(METRIC_DEFINITIONS))
        raise SoakError(f"metric set mismatch: missing={missing}, extra={extra}")
    metrics: list[dict[str, Any]] = []
    for name in sorted(values):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SoakError(f"metric {name} is not numeric: {value!r}")
        if not math.isfinite(value) or value < 0:
            raise SoakError(f"metric {name} is not finite and nonnegative: {value!r}")
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


def execute(
    model_path: Path,
    seed: int,
    minimum_duration_seconds: float,
    memory_growth_limit_bytes: int,
) -> tuple[list[dict[str, Any]], str | None]:
    started = time.monotonic()
    deadline = started + minimum_duration_seconds + SETUP_DEADLINE_SECONDS
    binary, binary_hash, build_seconds = mixed.build_binary(deadline)
    mixed.trace(
        "soak_binary_built",
        build_seconds=build_seconds,
        path=str(binary.relative_to(ROOT)),
        sha256=binary_hash,
    )
    port = mixed.free_loopback_port()
    run_dir = ROOT / ".qualification/serving" / f"soak-{os.getpid()}"
    adapter_dir = run_dir / "adapters"
    snapshot_dir = run_dir / "model-snapshots"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    environment = mixed.server_environment(
        RUNTIME_VARIANT, model_path, port, adapter_dir, snapshot_dir
    )
    environment["KILN_ROCM_GRAPH_CACHE_MAX"] = str(GRAPH_CACHE_MAX)
    process = subprocess.Popen(
        [str(binary), "--config", "/dev/null", "serve", "--served-model-id", mixed.MODEL_ID],
        cwd=ROOT,
        env=environment,
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
    shutdown: mixed.ShutdownOutcome | None = None
    snapshot_residue: list[str] = []
    values: dict[str, float | int] | None = None
    failures: list[str] = []
    stabilization_requests = 0
    stabilization_cancellations = 0
    stabilization_cycles = 0
    stabilization_stable_cycles = 0
    stabilization_final_gpu_delta = 0
    stabilization_final_rss_delta = 0
    stabilization_max_gpu_delta = 0
    stabilization_max_rss_delta = 0
    try:
        mixed.wait_ready(port, process, server_log, deadline)
        health_startup = mixed.read_stable_health(port, deadline, "soak startup health")
        debug_start = mixed.json_request(port, "GET", "/v1/debug/model-state")
        failures.extend(mixed.attest_runtime(RUNTIME_VARIANT, health_startup, debug_start))
        warmup: mixed.StreamResult | None = None
        health_start: dict[str, Any] | None = None
        for attempt in range(mixed.MAX_WARMUP_REQUESTS):
            warmup = mixed.run_stream(
                port,
                name=f"soak-warmup-{attempt + 1}",
                marker=mixed.workload_marker(seed, f"soak-warmup-{attempt + 1}"),
                prompt_words=16 + attempt * 8,
                max_tokens=mixed.WARMUP_MAX_TOKENS,
                seed=seed + attempt,
                absolute_deadline=deadline,
            )
            if not warmup.success:
                raise SoakError(f"warmup failed: {warmup.error or warmup.finish_reason}")
            health_start = wait_drained(port, deadline, "soak warmup")
            graph = mixed.graph_snapshot(health_start)
            if (
                graph["capture_successes"] >= 1
                and graph["replay_successes"] >= 1
                and graph["failures"] == 0
            ):
                break
        else:
            raise SoakError("graph warmup did not capture and replay")
        assert warmup is not None and health_start is not None

        steady_state_warmup_requests = 0
        steady_state_warmup_waves = 0
        prefix_warm = prefix_cache_snapshot(health_start)
        while prefix_warm["cached_entries"] < prefix_warm["max_entries"]:
            if steady_state_warmup_waves >= MAX_STEADY_STATE_WARMUP_WAVES:
                raise SoakError(
                    "prefix cache did not reach steady-state capacity within "
                    f"{MAX_STEADY_STATE_WARMUP_WAVES} warmup waves"
                )
            warm_results = run_wave(
                port,
                wave=steady_state_warmup_waves,
                base_seed=seed + 1_000_000,
                deadline=deadline,
                phase="warmup",
                prompt_epoch=steady_state_warmup_waves,
            )
            bad_warm = [
                result
                for result in warm_results
                if not result.success
                or result.finish_reason != "length"
                or result.completion_tokens != MAX_TOKENS
            ]
            if bad_warm:
                raise SoakError(
                    "steady-state warmup produced invalid responses: "
                    + ", ".join(
                        f"{item.name}({item.error or item.finish_reason},"
                        f"tokens={item.completion_tokens})"
                        for item in bad_warm[:8]
                    )
                )
            health_start = wait_drained(
                port, deadline, f"steady-state warmup {steady_state_warmup_waves}"
            )
            graph_warm = mixed.graph_snapshot(health_start)
            batching_warm = mixed.batching_snapshot(health_start)
            prefix_warm = prefix_cache_snapshot(health_start)
            warm_failures = mixed.attest_runtime(
                RUNTIME_VARIANT,
                health_start,
                mixed.json_request(port, "GET", "/v1/debug/model-state"),
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
            if warm_failures:
                raise SoakError("; ".join(warm_failures))
            steady_state_warmup_requests += len(warm_results)
            steady_state_warmup_waves += 1
            mixed.trace(
                "soak_steady_state_warmup",
                cached_entries=prefix_warm["cached_entries"],
                max_entries=prefix_warm["max_entries"],
                requests=steady_state_warmup_requests,
                waves=steady_state_warmup_waves,
            )

        previous_gpu = gpu_memory_bytes(port)
        previous_rss = rss_bytes(process.pid)
        stabilization_started = time.monotonic()
        while stabilization_cycles < MAX_STABILIZATION_CYCLES:
            cycle_failures: list[str] = []
            for offset in range(len(WAVE_CONCURRENCY)):
                stabilization_wave = (
                    stabilization_cycles * len(WAVE_CONCURRENCY) + offset
                )
                stable_results = run_wave(
                    port,
                    wave=stabilization_wave,
                    base_seed=seed,
                    deadline=deadline,
                    phase="stabilize",
                )
                stabilization_requests += len(stable_results)
                bad_stable = [
                    result
                    for result in stable_results
                    if not result.success
                    or result.finish_reason != "length"
                    or result.completion_tokens != MAX_TOKENS
                ]
                if bad_stable:
                    cycle_failures.append(
                        "stabilization produced invalid responses: "
                        + ", ".join(item.name for item in bad_stable[:8])
                    )
                if (stabilization_wave + 1) % CANCEL_EVERY_WAVES == 0:
                    cancellation_failure = run_cancellation(
                        port,
                        wave=stabilization_wave,
                        base_seed=seed + 2_000_000,
                        phase="stabilize",
                        deadline=deadline,
                    )
                    if cancellation_failure is None:
                        stabilization_cancellations += 1
                    else:
                        cycle_failures.append(cancellation_failure)

                health_start = wait_drained(
                    port, deadline, f"stabilization wave {stabilization_wave}"
                )
                graph_stable = mixed.graph_snapshot(health_start)
                batching_stable = mixed.batching_snapshot(health_start)
                prefix_stable = prefix_cache_snapshot(health_start)
                cycle_failures.extend(
                    mixed.attest_runtime(
                        RUNTIME_VARIANT,
                        health_start,
                        mixed.json_request(port, "GET", "/v1/debug/model-state"),
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
                    graph_stable["captured_graph_count"] > GRAPH_CACHE_MAX
                    or graph_stable["graph_slot_count"] > GRAPH_CACHE_MAX
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
                if (
                    prefix_stable["cached_entries"] != prefix_stable["max_entries"]
                    or prefix_stable["cached_state_bytes"]
                    != prefix_stable["max_state_bytes"]
                ):
                    cycle_failures.append(
                        "stabilization lost full prefix-cache residency"
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
            current_gpu = gpu_memory_bytes(port)
            current_rss = rss_bytes(process.pid)
            gpu_delta = max(0, current_gpu - previous_gpu)
            rss_delta = max(0, current_rss - previous_rss)
            stabilization_final_gpu_delta = gpu_delta
            stabilization_final_rss_delta = rss_delta
            stabilization_max_gpu_delta = max(stabilization_max_gpu_delta, gpu_delta)
            stabilization_max_rss_delta = max(stabilization_max_rss_delta, rss_delta)
            if (
                gpu_delta <= STABILIZATION_GPU_DELTA_LIMIT_BYTES
                and rss_delta <= STABILIZATION_RSS_DELTA_LIMIT_BYTES
            ):
                stabilization_stable_cycles += 1
            else:
                stabilization_stable_cycles = 0
            stabilization_cycles += 1
            mixed.trace(
                "soak_stabilization_cycle",
                cycle=stabilization_cycles,
                gpu_delta_bytes=gpu_delta,
                rss_delta_bytes=rss_delta,
                stable_cycles=stabilization_stable_cycles,
            )
            previous_gpu = current_gpu
            previous_rss = current_rss
            if (
                stabilization_cycles >= MIN_STABILIZATION_CYCLES
                and stabilization_stable_cycles >= REQUIRED_STABLE_CYCLES
            ):
                break
        else:
            raise SoakError(
                "GPU/RSS memory did not stabilize within "
                f"{MAX_STABILIZATION_CYCLES} cycles"
            )

        graph_start = mixed.graph_snapshot(health_start)
        batching_start = mixed.batching_snapshot(health_start)
        prefix_start = prefix_cache_snapshot(health_start)
        gpu_start = gpu_memory_bytes(port)
        rss_start = rss_bytes(process.pid)
        sampler.start()
        measurement_started = time.monotonic()
        all_results: list[mixed.StreamResult] = []
        rss_samples = [rss_start]
        wave = 0
        cancellations = 0

        while wave == 0 or time.monotonic() - measurement_started < minimum_duration_seconds:
            wave_failures: list[str] = []
            wave_results = run_wave(
                port, wave=wave, base_seed=seed, deadline=deadline
            )
            all_results.extend(wave_results)
            bad = [
                result
                for result in wave_results
                if not result.success
                or result.finish_reason != "length"
                or result.completion_tokens != MAX_TOKENS
            ]
            if bad:
                wave_failures.append(
                    "wave produced invalid responses: "
                    + ", ".join(
                        f"{item.name}({item.error or item.finish_reason},"
                        f"tokens={item.completion_tokens})"
                        for item in bad[:8]
                    )
                )

            if (wave + 1) % CANCEL_EVERY_WAVES == 0:
                cancellation_failure = run_cancellation(
                    port,
                    wave=wave,
                    base_seed=seed,
                    phase="measured",
                    deadline=deadline,
                )
                if cancellation_failure is not None:
                    wave_failures.append(cancellation_failure)
                else:
                    cancellations += 1

            health = wait_drained(port, deadline, f"soak wave {wave}")
            debug = mixed.json_request(port, "GET", "/v1/debug/model-state")
            wave_failures.extend(
                mixed.attest_runtime(RUNTIME_VARIANT, health, debug)
            )
            graph = mixed.graph_snapshot(health)
            batching = mixed.batching_snapshot(health)
            prefix = prefix_cache_snapshot(health)
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
                graph["captured_graph_count"] > GRAPH_CACHE_MAX
                or graph["graph_slot_count"] > GRAPH_CACHE_MAX
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

            current_gpu = gpu_memory_bytes(port)
            current_rss = rss_bytes(process.pid)
            rss_samples.append(current_rss)
            if current_gpu > gpu_start + memory_growth_limit_bytes:
                wave_failures.append(
                    f"GPU memory grew by {current_gpu - gpu_start} bytes after warmup"
                )
            if current_rss > rss_start + memory_growth_limit_bytes:
                wave_failures.append(
                    f"RSS grew by {current_rss - rss_start} bytes after warmup"
                )
            mixed.trace(
                "soak_wave_complete",
                cancellations=cancellations,
                elapsed_seconds=time.monotonic() - measurement_started,
                gpu_memory_bytes=current_gpu,
                requests=len(wave_results),
                rss_bytes=current_rss,
                wave=wave,
            )
            wave += 1
            if wave_failures:
                failures.extend(wave_failures)
                break

        sampler.close()
        health_end = wait_drained(port, deadline, "soak final health")
        graph_end = mixed.graph_snapshot(health_end)
        batching_end = mixed.batching_snapshot(health_end)
        prefix_end = prefix_cache_snapshot(health_end)
        failures.extend(mixed.attest_runtime_execution(RUNTIME_VARIANT, health_start, health_end))
        events = server_log.events_since(measurement_started)
        successes = [result for result in all_results if result.success]
        attributed, unexplained = mixed.classify_itl_outliers(
            warmup.itl_ms, successes, events
        )
        all_server_events = server_log.events_since(started)
        gpu_end = gpu_memory_bytes(port)
        rss_end = rss_bytes(process.pid)
        gpu_peak = max([gpu_start, gpu_end, *sampler.samples])
        rss_peak = max([rss_start, rss_end, *rss_samples])
        sync_values = mixed.external_yield_sync_metric_values(health_start, health_end)
        # Fallback stats are flattened by graph_snapshot with a `fallback_` prefix.
        fallback_delta = mixed.counter_delta(graph_start, graph_end, "fallback_total")
        request_failures = len(all_results) - len(successes)
        zero_tokens = sum(result.completion_tokens == 0 for result in all_results)
        non_finite = sum(
            result.error is not None and "non-finite" in result.error.lower()
            for result in all_results
        )
        duration = time.monotonic() - measurement_started
        itls = [gap for result in successes for gap in result.itl_ms]
        ttfts = [result.ttft_ms for result in successes]
        prompt_tokens = [result.prompt_tokens for result in successes]
        values = {
            "attributed_itl_outlier_count": attributed,
            "batching_error_count": mixed.counter_delta(
                batching_start, batching_end, "total_errors"
            ),
            "batching_max_observed_active_requests": batching_end[
                "max_observed_active_requests"
            ],
            "batching_max_observed_batch_size": batching_end["max_observed_batch_size"],
            "cancellation_confirmed_count": cancellations,
            "completion_token_count": sum(result.completion_tokens for result in successes),
            "device_fault_event_count": sum(
                event.category == "device_fault" for event in all_server_events
            ),
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
            "itl_ms_p50": mixed.percentile_r7(itls, 0.5),
            "itl_ms_p99": mixed.percentile_r7(itls, 0.99),
            "itl_ms_p999": mixed.percentile_r7(itls, 0.999),
            "kv_blocks_end": batching_end["blocks_total"],
            "kv_blocks_start": batching_start["blocks_total"],
            "kv_blocks_used_end": batching_end["blocks_used"],
            "kv_unaccounted_blocks_end": unaccounted_blocks(
                batching_end, prefix_end
            ),
            "non_finite_response_count": non_finite,
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
            "prompt_tokens_max": max(prompt_tokens, default=0),
            "prompt_tokens_min": min(prompt_tokens, default=0),
            "request_failure_count": request_failures,
            "request_count": len(all_results),
            "rss_baseline_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "rss_growth_bytes": max(0, rss_end - rss_start),
            "rss_peak_bytes": rss_peak,
            "shutdown_forced_count": 0,
            "shutdown_nonzero_count": 0,
            "snapshot_residue_count": 0,
            "soak_duration_seconds": duration,
            "steady_state_warmup_request_count": steady_state_warmup_requests,
            "steady_state_warmup_wave_count": steady_state_warmup_waves,
            "stabilization_cancellation_count": stabilization_cancellations,
            "stabilization_cycle_count": stabilization_cycles,
            "stabilization_final_gpu_delta_bytes": stabilization_final_gpu_delta,
            "stabilization_final_rss_delta_bytes": stabilization_final_rss_delta,
            "stabilization_max_gpu_delta_bytes": stabilization_max_gpu_delta,
            "stabilization_max_rss_delta_bytes": stabilization_max_rss_delta,
            "stabilization_request_count": stabilization_requests,
            "stabilization_stable_cycle_count": stabilization_stable_cycles,
            "ttft_ms_p50": mixed.percentile_r7(ttfts, 0.5),
            "ttft_ms_p99": mixed.percentile_r7(ttfts, 0.99),
            "ttft_ms_p999": mixed.percentile_r7(ttfts, 0.999),
            "unexplained_itl_outlier_count": unexplained,
            "wave_count": wave,
            "zero_token_response_count": zero_tokens,
        }
        if duration < minimum_duration_seconds:
            failures.append(
                f"soak duration {duration:.3f}s was below {minimum_duration_seconds:.3f}s"
            )
        if request_failures != 0 or zero_tokens != 0:
            failures.append(
                f"soak had request_failures={request_failures}, zero_tokens={zero_tokens}"
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
        if values["graph_replay_success_count"] < 1:
            failures.append("soak completed without a measured graph replay")
        if values["graph_slot_reuse_count"] < 1:
            failures.append("soak completed without measured graph-slot reuse")
        if values["graph_retained_count_end"] > GRAPH_CACHE_MAX:
            failures.append("retained graph residency exceeded the graph cache bound")
        if values["graph_slot_count_end"] > GRAPH_CACHE_MAX:
            failures.append("graph-slot residency exceeded the graph cache bound")
        if values["graph_slot_idle_count_end"] != values["graph_slot_count_end"]:
            failures.append("not every retained graph slot was idle at final drain")
        if values["prefix_cache_lookup_hit_count"] < 1:
            failures.append("soak completed without a measured prefix-cache hit")
        if values["prefix_cache_hit_blocks"] < 1:
            failures.append("soak completed without reusing a cached KV block")
        if prefix_start["cached_entries"] != prefix_start["max_entries"]:
            failures.append("measurement began before the prefix cache reached capacity")
        if prefix_end["cached_entries"] != prefix_start["cached_entries"]:
            failures.append("prefix-cache entry residency changed during measured soak")
        if prefix_end["cached_state_bytes"] != prefix_start["cached_state_bytes"]:
            failures.append("prefix-cache state residency changed during measured soak")
        if batching_end["blocks_total"] != batching_start["blocks_total"]:
            failures.append("KV block capacity changed during soak")
        if sampler.errors:
            failures.append("GPU memory sampler errors: " + ", ".join(sampler.errors))
        if gpu_peak > gpu_start + memory_growth_limit_bytes:
            failures.append("peak GPU memory exceeded the post-warmup growth limit")
        if rss_peak > rss_start + memory_growth_limit_bytes:
            failures.append("peak RSS exceeded the post-warmup growth limit")
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        sampler.close()
        shutdown = mixed.terminate_process(process)
        server_log.join()
        snapshot_residue = mixed.snapshot_payload_residue(snapshot_dir)
        shutil.rmtree(run_dir, ignore_errors=True)

    if values is None:
        values = {name: 0 for name in METRIC_DEFINITIONS}
        values["soak_duration_seconds"] = max(0.0, time.monotonic() - started)
        values["stabilization_cancellation_count"] = stabilization_cancellations
        values["stabilization_cycle_count"] = stabilization_cycles
        values["stabilization_final_gpu_delta_bytes"] = stabilization_final_gpu_delta
        values["stabilization_final_rss_delta_bytes"] = stabilization_final_rss_delta
        values["stabilization_max_gpu_delta_bytes"] = stabilization_max_gpu_delta
        values["stabilization_max_rss_delta_bytes"] = stabilization_max_rss_delta
        values["stabilization_request_count"] = stabilization_requests
        values["stabilization_stable_cycle_count"] = stabilization_stable_cycles
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
    details = " | ".join(dict.fromkeys(failures)) if failures else None
    return metrics_from_values(values), mixed.bounded_details(details)


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
    if variant != RUNTIME_VARIANT:
        print(
            f"{VARIANT_ENV} must be {RUNTIME_VARIANT!r}, got {variant!r}",
            file=sys.stderr,
        )
        return 2
    if not result_path_value:
        print(f"{RESULT_ENV} is required", file=sys.stderr)
        return 2
    result_path = Path(result_path_value)
    status = "failed"
    details: str | None = None
    metrics = metrics_from_values({name: 0 for name in METRIC_DEFINITIONS})
    try:
        model_path = args.model_path.resolve(strict=True)
        if not model_path.is_dir():
            raise SoakError("--model-path must be a directory")
        metrics, details = execute(
            model_path,
            args.seed,
            args.minimum_duration_seconds,
            args.memory_growth_limit_bytes,
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
            args.minimum_duration_seconds, args.memory_growth_limit_bytes
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
