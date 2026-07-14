#!/usr/bin/env python3
"""Qualify ROCm graph byte budgets and concurrency without changing binaries."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import serve_mixed_load as mixed


ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "rocm-graph-budget-concurrency"
VARIANT_ID = "headroom-vs-tight-budget"
RESULT_ENV = mixed.RESULT_ENV
VARIANT_ENV = mixed.VARIANT_ENV
CONCURRENCY_LEVELS = (1, 8, 16, 32, 64)
PROMPT_WORD_BUCKETS = (16, 32, 64, 128, 256, 384, 512, 768)
MAX_TOKENS = 8
WARMUP_MAX_TOKENS = 16
GRAPH_CACHE_ENTRIES = 64
HEADROOM_BUDGET_BYTES = 1 << 30
TIGHT_BUDGET_BYTES = 64 << 20
REQUEST_TIMEOUT_SECONDS = 300.0
SERVER_REQUEST_TIMEOUT_SECONDS = 360
OVERALL_TIMEOUT_SECONDS = 2100.0
ARM_ORDER = ("headroom", "tight")
ARM_BUDGETS = {
    "headroom": HEADROOM_BUDGET_BYTES,
    "tight": TIGHT_BUDGET_BYTES,
}


class ResilienceError(RuntimeError):
    pass


def _base_config() -> dict[str, Any]:
    value = mixed._variant_config(
        serving_profile="experimental",
        kv_autoscale_requested=False,
        kv_autoscale_enabled=False,
        memory_reclaim_requested_mode="off",
        memory_reclaim_mode="off",
        rocm_graphs_requested=True,
        rocm_graphs_enabled=True,
    )
    value["runtime"].update(
        {
            "rocm_graph_cache_entries": GRAPH_CACHE_ENTRIES,
            "rocm_graph_cache_max_bytes_by_arm": ARM_BUDGETS,
            "rocm_graph_mode": "lazy_capture_replay",
        }
    )
    value["server"]["request_timeout_seconds"] = SERVER_REQUEST_TIMEOUT_SECONDS
    value["workload"] = {
        "arm_order": {
            f"arm_{index}": arm for index, arm in enumerate(ARM_ORDER)
        },
        "build_reuse": "one_source_bound_binary_for_both_arms",
        "concurrency_levels": {
            f"level_{index}": level
            for index, level in enumerate(CONCURRENCY_LEVELS)
        },
        "graph_cache_entries": GRAPH_CACHE_ENTRIES,
        "headroom_budget_bytes": HEADROOM_BUDGET_BYTES,
        "max_tokens": MAX_TOKENS,
        "output_comparison": "exact_canonical_streamed_semantic_deltas",
        "overall_timeout_seconds": int(OVERALL_TIMEOUT_SECONDS),
        "pause_policy": "zero_attributed_or_unexplained_itl_outliers",
        "prompt_word_buckets": {
            f"bucket_{index}": words
            for index, words in enumerate(PROMPT_WORD_BUCKETS)
        },
        "request_count_per_arm": sum(CONCURRENCY_LEVELS),
        "request_timeout_seconds": int(REQUEST_TIMEOUT_SECONDS),
        "tight_budget_bytes": TIGHT_BUDGET_BYTES,
        "warmup_max_tokens": WARMUP_MAX_TOKENS,
    }
    return value


EFFECTIVE_CONFIG = _base_config()
mixed.VARIANT_CONFIGS[VARIANT_ID] = EFFECTIVE_CONFIG


METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    "attributed_itl_outlier_count": ("count", "sum", True),
    "case_failure_count": ("count", "sum", True),
    "graph_budget_event_count": ("count", "sum", False),
    "headroom_arm_completed": ("count", "sum", False),
    "headroom_arm_started": ("count", "sum", False),
    "headroom_attempted_request_count": ("count", "sum", False),
    "headroom_graph_capture_count": ("count", "sum", False),
    "headroom_graph_failure_count": ("count", "sum", True),
    "headroom_graph_peak_retained_bytes": ("bytes", "max", True),
    "headroom_graph_replay_count": ("count", "sum", False),
    "headroom_completed_request_count": ("count", "sum", False),
    "headroom_peak_gpu_memory_used_bytes": ("bytes", "max", True),
    "max_completed_concurrency": ("requests", "max", False),
    "output_mismatch_count": ("count", "sum", True),
    "request_failure_count": ("count", "sum", True),
    "tight_arm_completed": ("count", "sum", False),
    "tight_arm_started": ("count", "sum", False),
    "tight_attempted_request_count": ("count", "sum", False),
    "tight_completed_request_count": ("count", "sum", False),
    "tight_graph_budget_eviction_count": ("count", "sum", False),
    "tight_graph_byte_budget_rejection_count": ("count", "sum", False),
    "tight_graph_capture_count": ("count", "sum", False),
    "tight_graph_failure_count": ("count", "sum", True),
    "tight_graph_peak_retained_bytes": ("bytes", "max", True),
    "tight_graph_pre_capture_byte_budget_skip_count": ("count", "sum", False),
    "tight_graph_replay_count": ("count", "sum", False),
    "tight_peak_gpu_memory_used_bytes": ("bytes", "max", True),
    "unexplained_itl_outlier_count": ("count", "sum", True),
}
for _level in CONCURRENCY_LEVELS:
    METRIC_DEFINITIONS[f"concurrency_{_level}_attempted_request_count"] = (
        "count",
        "sum",
        False,
    )
    METRIC_DEFINITIONS[f"concurrency_{_level}_e2e_ms_p99"] = (
        "ms",
        "p99",
        True,
    )
    METRIC_DEFINITIONS[f"concurrency_{_level}_itl_ms_p99"] = (
        "ms",
        "p99",
        True,
    )
    METRIC_DEFINITIONS[f"concurrency_{_level}_failure_count"] = (
        "count",
        "sum",
        True,
    )
    METRIC_DEFINITIONS[f"concurrency_{_level}_request_count"] = (
        "count",
        "sum",
        False,
    )
    METRIC_DEFINITIONS[f"concurrency_{_level}_ttft_ms_p99"] = (
        "ms",
        "p99",
        True,
    )


@dataclasses.dataclass(frozen=True)
class ArmRun:
    name: str
    budget_bytes: int
    results: tuple[mixed.StreamResult, ...]
    outputs: dict[str, str]
    graph_start: dict[str, int]
    graph_end: dict[str, int]
    peak_gpu_memory_used_bytes: int
    attributed_itl_outliers: int
    unexplained_itl_outliers: int


@dataclasses.dataclass
class RunEvidence:
    arm_started: set[str] = dataclasses.field(default_factory=set)
    arm_completed: set[str] = dataclasses.field(default_factory=set)
    results: dict[str, list[mixed.StreamResult]] = dataclasses.field(
        default_factory=lambda: {arm: [] for arm in ARM_ORDER}
    )
    outputs_by_arm: dict[str, dict[str, str]] = dataclasses.field(
        default_factory=lambda: {arm: {} for arm in ARM_ORDER}
    )
    attempted_by_level: dict[int, int] = dataclasses.field(default_factory=dict)
    tight_attempted_by_level: dict[int, int] = dataclasses.field(default_factory=dict)
    failures_by_level: dict[int, int] = dataclasses.field(default_factory=dict)
    graph_by_arm: dict[str, dict[str, int]] = dataclasses.field(default_factory=dict)
    peak_gpu_memory_used_bytes: dict[str, int] = dataclasses.field(default_factory=dict)
    attributed_itl_outliers: dict[str, int] = dataclasses.field(default_factory=dict)
    unexplained_itl_outliers: dict[str, int] = dataclasses.field(default_factory=dict)
    max_completed_concurrency: int = 0
    request_failure_count: int = 0
    case_failure_count: int = 0

    def begin_wave(self, arm: str, level: int, request_count: int) -> None:
        if arm == "headroom":
            self.attempted_by_level[level] = request_count
        elif arm == "tight":
            self.tight_attempted_by_level[level] = request_count

    def record_wave(
        self,
        arm: str,
        level: int,
        wave: list[mixed.StreamResult],
    ) -> list[mixed.StreamResult]:
        successful = [result for result in wave if result.success]
        failures = len(wave) - len(successful)
        self.results[arm].extend(successful)
        self.outputs_by_arm[arm].update(
            {
                result.name: canonical_semantic_hash(result)
                for result in successful
            }
        )
        self.request_failure_count += failures
        if arm == "headroom":
            self.begin_wave(arm, level, len(wave))
            self.failures_by_level[level] = failures
            if failures == 0 and len(wave) == level:
                self.max_completed_concurrency = max(
                    self.max_completed_concurrency,
                    level,
                )
        else:
            self.begin_wave(arm, level, len(wave))
        return successful

    def record_graph(self, arm: str, value: dict[str, int]) -> None:
        self.graph_by_arm[arm] = dict(value)

    def record_peak_memory(self, arm: str, value: int) -> None:
        self.peak_gpu_memory_used_bytes[arm] = max(
            self.peak_gpu_memory_used_bytes.get(arm, 0),
            value,
        )


def canonical_semantic_hash(result: mixed.StreamResult) -> str:
    records: list[Any] = []
    for event in result.semantic_deltas:
        choices = event.get("choices")
        if not isinstance(choices, list):
            raise ResilienceError(f"{result.name} emitted a semantic event without choices")
        records.append(
            [
                {
                    "index": choice.get("index"),
                    "delta": choice.get("delta"),
                }
                for choice in choices
                if isinstance(choice, dict)
            ]
        )
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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


def run_wave(
    port: int,
    arm: str,
    level: int,
    seed: int,
    deadline: float,
) -> list[mixed.StreamResult]:
    abort = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=level)
    futures: list[concurrent.futures.Future[mixed.StreamResult]] = []
    try:
        for index in range(level):
            role = f"resilience-c{level}-r{index:02d}"
            futures.append(
                pool.submit(
                    mixed.run_stream,
                    port,
                    name=role,
                    marker=mixed.workload_marker(seed, role),
                    prompt_words=PROMPT_WORD_BUCKETS[index % len(PROMPT_WORD_BUCKETS)],
                    max_tokens=MAX_TOKENS,
                    seed=seed + level * 1000 + index,
                    absolute_deadline=deadline,
                    abort_event=abort,
                    request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
                )
            )
        results = [
            future.result(
                timeout=mixed.remaining_until(deadline, f"{arm} concurrency {level}")
            )
            for future in futures
        ]
    finally:
        abort.set()
        for future in futures:
            future.cancel()
        _, unfinished = concurrent.futures.wait(
            futures, timeout=max(0.0, min(10.0, deadline - time.monotonic()))
        )
        pool.shutdown(wait=False, cancel_futures=True)
        if unfinished:
            raise ResilienceError(
                f"{arm} concurrency {level} left {len(unfinished)} request workers"
            )
    return results


def run_arm(
    binary: Path,
    model_path: Path,
    seed: int,
    arm: str,
    deadline: float,
    evidence: RunEvidence,
) -> ArmRun:
    budget_bytes = ARM_BUDGETS[arm]
    port = mixed.free_loopback_port()
    run_dir = mixed.create_serving_run_dir(f"graph-resilience-{arm}")
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
        rocm_graph_mode="lazy_capture_replay",
        rocm_graph_cache_entries=GRAPH_CACHE_ENTRIES,
        rocm_graph_cache_max_bytes=budget_bytes,
    )
    process, server_log = mixed.start_server(binary, config_path, VARIANT_ID)
    sampler = mixed.MemorySampler(port)
    shutdown: mixed.ShutdownOutcome | None = None
    residue: list[str] = []
    measured_started: float | None = None
    warmup: mixed.StreamResult | None = None
    results: list[mixed.StreamResult] = []
    try:
        mixed.wait_ready(port, process, server_log, deadline)
        startup = wait_drained(port, deadline, f"{arm} startup")
        debug = mixed.json_request(port, "GET", "/v1/debug/model-state")
        failures = mixed.attest_runtime(
            VARIANT_ID,
            startup,
            debug,
            rocm_graph_cache_entries=GRAPH_CACHE_ENTRIES,
            rocm_graph_cache_max_bytes=budget_bytes,
        )
        if failures:
            raise ResilienceError("; ".join(failures))

        for index in range(8):
            warmup = mixed.run_stream(
                port,
                name=f"warmup-{index}",
                marker=mixed.workload_marker(seed, f"resilience-warmup-{index}"),
                prompt_words=PROMPT_WORD_BUCKETS[index % len(PROMPT_WORD_BUCKETS)],
                max_tokens=WARMUP_MAX_TOKENS,
                seed=seed + index,
                absolute_deadline=deadline,
                request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            )
            if not warmup.success:
                raise ResilienceError(f"{arm} warmup failed: {warmup.error}")
            warm_health = wait_drained(port, deadline, f"{arm} warmup {index}")
            warm_graph = mixed.graph_snapshot(warm_health)
            if arm == "headroom":
                ready = (
                    warm_graph["capture_successes"] > 0
                    and warm_graph["replay_successes"] > 0
                )
            else:
                ready = graph_budget_events(warm_graph) > 0 or (
                    warm_graph["capture_successes"] > 0
                    and warm_graph["replay_successes"] > 0
                )
            if ready:
                break
        else:
            raise ResilienceError(f"{arm} warmup established no graph execution outcome")
        assert warmup is not None

        health_start = wait_drained(port, deadline, f"{arm} measurement start")
        graph_start = mixed.graph_snapshot(health_start)
        evidence.record_graph(arm, graph_start)
        measured_started = time.monotonic()
        sampler.start()
        for level in CONCURRENCY_LEVELS:
            evidence.begin_wave(arm, level, level)
            wave = run_wave(port, arm, level, seed, deadline)
            failures = [result for result in wave if not result.success]
            results.extend(evidence.record_wave(arm, level, wave))
            if failures:
                raise ResilienceError(
                    f"{arm} concurrency {level} had {len(failures)} failed requests: "
                    + "; ".join(result.error or result.finish_reason or "unknown" for result in failures[:3])
                )
            for result in wave:
                if result.finish_reason != "length" or result.completion_tokens != MAX_TOKENS:
                    raise ResilienceError(
                        f"{arm} {result.name} did not preserve the fixed output denominator"
                    )
            drained = wait_drained(port, deadline, f"{arm} concurrency {level}")
            evidence.record_graph(arm, mixed.graph_snapshot(drained))
        sampler.close()

        health_end = wait_drained(port, deadline, f"{arm} final")
        debug_end = mixed.json_request(port, "GET", "/v1/debug/model-state")
        failures = mixed.attest_runtime(
            VARIANT_ID,
            health_end,
            debug_end,
            rocm_graph_cache_entries=GRAPH_CACHE_ENTRIES,
            rocm_graph_cache_max_bytes=budget_bytes,
        )
        if failures:
            raise ResilienceError("; ".join(failures))
        graph_end = mixed.graph_snapshot(health_end)
        evidence.record_graph(arm, graph_end)
        evidence.record_peak_memory(arm, max(sampler.samples, default=0))
        if process.poll() is not None:
            raise ResilienceError(f"{arm} server exited during measurement")
        if graph_end["failures"] != 0:
            raise ResilienceError(f"{arm} recorded {graph_end['failures']} graph failures")
        if graph_end["retained_bytes"] > budget_bytes:
            raise ResilienceError(f"{arm} retained graph bytes exceed the configured budget")
        if graph_end["captured_graph_count"] > GRAPH_CACHE_ENTRIES:
            raise ResilienceError(f"{arm} retained graph entries exceed the configured limit")
        if arm == "headroom" and (
            graph_end["capture_successes"] == 0 or graph_end["replay_successes"] == 0
        ):
            raise ResilienceError("headroom arm did not capture and replay a graph")
        if arm == "tight" and graph_budget_events(graph_end) == 0:
            raise ResilienceError(
                "tight arm never exercised a byte-budget skip, rejection, eviction, or fallback"
            )

        events = server_log.events_since(measured_started)
        attributed, unexplained = mixed.classify_itl_outliers(
            warmup.itl_ms, results, events
        )
        evidence.attributed_itl_outliers[arm] = attributed
        evidence.unexplained_itl_outliers[arm] = unexplained
        if attributed or unexplained:
            raise ResilienceError(
                f"{arm} observed {attributed} attributed and {unexplained} unexplained ITL outliers"
            )
        device_faults = [event for event in events if event.category == "device_fault"]
        if device_faults:
            raise ResilienceError(f"{arm} observed {len(device_faults)} device faults")

        return ArmRun(
            name=arm,
            budget_bytes=budget_bytes,
            results=tuple(results),
            outputs={result.name: canonical_semantic_hash(result) for result in results},
            graph_start=graph_start,
            graph_end=graph_end,
            peak_gpu_memory_used_bytes=max(sampler.samples, default=0),
            attributed_itl_outliers=attributed,
            unexplained_itl_outliers=unexplained,
        )
    finally:
        sampler.close()
        evidence.record_peak_memory(arm, max(sampler.samples, default=0))
        if process.poll() is None:
            try:
                evidence.record_graph(
                    arm,
                    mixed.graph_snapshot(mixed.json_request(port, "GET", "/health")),
                )
            except Exception:
                pass
        shutdown = mixed.terminate_process(process)
        server_log.join()
        if measured_started is not None and warmup is not None:
            events = server_log.events_since(measured_started)
            try:
                attributed, unexplained = mixed.classify_itl_outliers(
                    warmup.itl_ms,
                    results,
                    events,
                )
                evidence.attributed_itl_outliers[arm] = attributed
                evidence.unexplained_itl_outliers[arm] = unexplained
            except Exception:
                pass
        residue = mixed.snapshot_payload_residue(snapshot_dir)
        shutil.rmtree(run_dir, ignore_errors=True)
        if shutdown.forced or shutdown.returncode != 0 or residue:
            raise ResilienceError(
                f"{arm} teardown failed: forced={shutdown.forced}, "
                f"returncode={shutdown.returncode}, residue={residue}"
            )


def graph_budget_events(graph: dict[str, int]) -> int:
    return sum(
        graph.get(field, 0)
        for field in (
            "budget_evictions",
            "byte_budget_rejections",
            "pre_capture_byte_budget_skips",
            "fallback_graph_cache_byte_budget",
        )
    )


def percentile(values: list[float], probability: float) -> float:
    return mixed.percentile_r7(values, probability) if values else 0.0


def metrics_from_evidence(
    evidence: RunEvidence,
) -> tuple[list[dict[str, Any]], str | None]:
    headroom_results = evidence.results["headroom"]
    headroom_outputs = evidence.outputs_by_arm["headroom"]
    tight_outputs = evidence.outputs_by_arm["tight"]
    output_mismatches = 0
    if set(ARM_ORDER).issubset(evidence.arm_completed):
        common_outputs = set(headroom_outputs) & set(tight_outputs)
        output_mismatches = sum(
            headroom_outputs[name] != tight_outputs[name] for name in common_outputs
        ) + len(set(headroom_outputs) ^ set(tight_outputs))

    headroom_graph = evidence.graph_by_arm.get("headroom", {})
    tight_graph = evidence.graph_by_arm.get("tight", {})
    headroom_failures = {
        level: max(
            evidence.failures_by_level.get(level, 0),
            evidence.attempted_by_level.get(level, 0)
            - sum(
                result.name.startswith(f"resilience-c{level}-")
                for result in headroom_results
            ),
        )
        for level in CONCURRENCY_LEVELS
    }
    tight_failure_count = max(
        0,
        sum(evidence.tight_attempted_by_level.values())
        - len(evidence.results["tight"]),
    )
    values: dict[str, float | int] = {
        "attributed_itl_outlier_count": sum(evidence.attributed_itl_outliers.values()),
        "case_failure_count": evidence.case_failure_count,
        "graph_budget_event_count": graph_budget_events(tight_graph),
        "headroom_arm_completed": int("headroom" in evidence.arm_completed),
        "headroom_arm_started": int("headroom" in evidence.arm_started),
        "headroom_attempted_request_count": sum(evidence.attempted_by_level.values()),
        "headroom_completed_request_count": len(headroom_results),
        "headroom_graph_capture_count": headroom_graph.get("capture_successes", 0),
        "headroom_graph_failure_count": headroom_graph.get("failures", 0),
        "headroom_graph_peak_retained_bytes": headroom_graph.get(
            "peak_retained_bytes", 0
        ),
        "headroom_graph_replay_count": headroom_graph.get("replay_successes", 0),
        "headroom_peak_gpu_memory_used_bytes": evidence.peak_gpu_memory_used_bytes.get(
            "headroom", 0
        ),
        "max_completed_concurrency": evidence.max_completed_concurrency,
        "output_mismatch_count": output_mismatches,
        "request_failure_count": max(
            evidence.request_failure_count,
            sum(headroom_failures.values()) + tight_failure_count,
        ),
        "tight_arm_completed": int("tight" in evidence.arm_completed),
        "tight_arm_started": int("tight" in evidence.arm_started),
        "tight_attempted_request_count": sum(
            evidence.tight_attempted_by_level.values()
        ),
        "tight_completed_request_count": len(evidence.results["tight"]),
        "tight_graph_budget_eviction_count": tight_graph.get("budget_evictions", 0),
        "tight_graph_byte_budget_rejection_count": tight_graph.get(
            "byte_budget_rejections", 0
        ),
        "tight_graph_capture_count": tight_graph.get("capture_successes", 0),
        "tight_graph_failure_count": tight_graph.get("failures", 0),
        "tight_graph_peak_retained_bytes": tight_graph.get("peak_retained_bytes", 0),
        "tight_graph_pre_capture_byte_budget_skip_count": tight_graph.get(
            "pre_capture_byte_budget_skips", 0
        ),
        "tight_graph_replay_count": tight_graph.get("replay_successes", 0),
        "tight_peak_gpu_memory_used_bytes": evidence.peak_gpu_memory_used_bytes.get(
            "tight", 0
        ),
        "unexplained_itl_outlier_count": sum(
            evidence.unexplained_itl_outliers.values()
        ),
    }
    for level in CONCURRENCY_LEVELS:
        selected = [
            result
            for result in headroom_results
            if result.name.startswith(f"resilience-c{level}-")
        ]
        values[f"concurrency_{level}_attempted_request_count"] = (
            evidence.attempted_by_level.get(level, 0)
        )
        values[f"concurrency_{level}_e2e_ms_p99"] = percentile(
            [result.e2e_ms for result in selected], 0.99
        )
        values[f"concurrency_{level}_failure_count"] = headroom_failures[level]
        values[f"concurrency_{level}_itl_ms_p99"] = percentile(
            [gap for result in selected for gap in result.itl_ms], 0.99
        )
        values[f"concurrency_{level}_request_count"] = len(selected)
        values[f"concurrency_{level}_ttft_ms_p99"] = percentile(
            [result.ttft_ms for result in selected], 0.99
        )
    details = None if output_mismatches == 0 else f"{output_mismatches} exact output mismatches"
    metrics = []
    for name in sorted(METRIC_DEFINITIONS):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ResilienceError(f"metric {name} is not finite: {value!r}")
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
    return metrics, details


def metrics_from_arms(arms: dict[str, ArmRun]) -> tuple[list[dict[str, Any]], str | None]:
    evidence = RunEvidence()
    for arm_name, arm in arms.items():
        evidence.arm_started.add(arm_name)
        evidence.arm_completed.add(arm_name)
        evidence.results[arm_name].extend(arm.results)
        evidence.outputs_by_arm[arm_name].update(arm.outputs)
        evidence.record_graph(arm_name, arm.graph_end)
        evidence.record_peak_memory(arm_name, arm.peak_gpu_memory_used_bytes)
        evidence.attributed_itl_outliers[arm_name] = arm.attributed_itl_outliers
        evidence.unexplained_itl_outliers[arm_name] = arm.unexplained_itl_outliers
    for level in CONCURRENCY_LEVELS:
        selected = [
            result
            for result in evidence.results["headroom"]
            if result.name.startswith(f"resilience-c{level}-")
        ]
        evidence.attempted_by_level[level] = len(selected)
        evidence.failures_by_level[level] = 0
        evidence.tight_attempted_by_level[level] = len(
            [
                result
                for result in evidence.results["tight"]
                if result.name.startswith(f"resilience-c{level}-")
            ]
        )
    evidence.max_completed_concurrency = max(CONCURRENCY_LEVELS)
    return metrics_from_evidence(evidence)


def zero_metrics() -> list[dict[str, Any]]:
    return metrics_from_evidence(RunEvidence())[0]


def write_result(path: Path, value: dict[str, Any]) -> None:
    mixed.write_result(path, value)


def execute(model_path: Path, seed: int, evidence: RunEvidence) -> None:
    deadline = time.monotonic() + OVERALL_TIMEOUT_SECONDS
    binary, binary_hash, build_seconds = mixed.build_binary(deadline)
    mixed.trace(
        "graph_resilience_binary_built",
        build_seconds=build_seconds,
        path=str(binary.relative_to(ROOT)),
        sha256=binary_hash,
    )
    for arm in ARM_ORDER:
        evidence.arm_started.add(arm)
        run_arm(binary, model_path, seed, arm, deadline, evidence)
        evidence.arm_completed.add(arm)


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
        print(f"{RESULT_ENV} is required", file=os.sys.stderr)
        return 2
    status = "failed"
    details: str | None = None
    evidence = RunEvidence()
    metrics = zero_metrics()
    try:
        if variant != VARIANT_ID:
            raise ResilienceError(f"{VARIANT_ENV} must be {VARIANT_ID!r}, got {variant!r}")
        model_path = args.model_path.resolve(strict=True)
        if not model_path.is_dir():
            raise ResilienceError("--model-path must be a directory")
        execute(model_path, args.seed, evidence)
        metrics, details = metrics_from_evidence(evidence)
        status = "passed" if details is None else "failed"
        if status == "failed":
            evidence.case_failure_count = 1
            metrics, _ = metrics_from_evidence(evidence)
    except Exception as exc:
        details = f"{type(exc).__name__}: {exc}"
        evidence.case_failure_count = 1
        metrics, _ = metrics_from_evidence(evidence)
        mixed.trace("graph_resilience_error", details=details)
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
        write_result(Path(result_path_value), result)
    except Exception as exc:
        print(f"cannot write qualification result: {exc}", file=os.sys.stderr)
        return 2
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
