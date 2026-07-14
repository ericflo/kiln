#!/usr/bin/env python3
"""Qualify legacy and stream-ordered ROCm synchronization with one bounded A/B."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import serve_mixed_load as mixed
import serve_rocm_graph_correctness as correctness


ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "rocm-synchronization-ab"
VARIANT_ID = "legacy-vs-stream-ordered"
RESULT_ENV = mixed.RESULT_ENV
VARIANT_ENV = mixed.VARIANT_ENV
MODES = ("legacy_host_barriers", "stream_ordered")
WAVES = (
    ("single", (16,)),
    ("pair", (32, 96)),
    ("four-way", (16, 64, 192, 384)),
)
MAX_TOKENS = 32
WARMUP_MAX_TOKENS = 16
REQUEST_TIMEOUT_SECONDS = 90.0
OVERALL_TIMEOUT_SECONDS = 1500.0
STALL_THRESHOLD_MS = 250.0
PAUSE_GATE_MS = 2_000.0
ROCM_SYNC_REASONS = (
    "explicit_device_drain",
    "explicit_stream_drain",
    "tensor_handoff",
    "external_yield",
    "activation_output",
    "elementwise_output",
    "cast_output",
    "concat_input",
    "concat_output",
    "contiguous_output",
    "repeat_heads_output",
    "matmul_output",
    "matmul_cast_boundary",
    "in_place_mutation",
    "memory_reclaim",
    "graph_boundary",
    "full_attention_handoff",
    "model_handoff",
    "host_readback",
    "allocation_lifetime",
    "error_recovery",
    "global_state_mutation",
    "capture_rollback",
)


class SynchronizationQualificationError(RuntimeError):
    pass


def mode_config(mode: str) -> dict[str, Any]:
    value = mixed._variant_config(
        serving_profile="experimental",
        kv_autoscale_requested=False,
        kv_autoscale_enabled=False,
        memory_reclaim_requested_mode="off",
        memory_reclaim_mode="off",
        rocm_graphs_requested=False,
        rocm_graphs_enabled=False,
    )
    value["runtime"]["rocm_synchronization_mode"] = mode
    value["runtime"]["rocm_graph_mode"] = "disabled"
    value["workload"] = {
        "arm_order": {
            "arm_0": "legacy_host_barriers",
            "arm_1": "stream_ordered",
        },
        "build_reuse": "one_source_bound_binary_for_both_arms",
        "comparison": "same_binary_same_seed_fixed_output",
        "correctness_probe": {
            "comparison": "exact_action_tokens_and_selected_logprobs",
            "max_tokens": 16,
            "prompt_words": 64,
            "sampling": "temperature_1_top_k_16_fixed_seed",
        },
        "expected_completion_tokens_per_arm": sum(
            len(prompt_words) for _, prompt_words in WAVES
        )
        * MAX_TOKENS,
        "graph_mode": "disabled",
        "max_tokens": MAX_TOKENS,
        "overall_timeout_seconds": int(OVERALL_TIMEOUT_SECONDS),
        "pause_gate_ms": int(PAUSE_GATE_MS),
        "reason_dimension_count": len(ROCM_SYNC_REASONS),
        "request_count_per_arm": sum(len(prompt_words) for _, prompt_words in WAVES),
        "request_timeout_seconds": int(REQUEST_TIMEOUT_SECONDS),
        "stall_evidence_threshold_ms": int(STALL_THRESHOLD_MS),
        "telemetry_checkpoints": "startup_post_warmup_after_each_wave_final",
        "teardown_grace_seconds_per_arm": int(
            mixed.SERVER_SHUTDOWN_GRACE_SECONDS
        ),
        "warmup_max_tokens": WARMUP_MAX_TOKENS,
        "warmup_prompt_words": 16,
        "waves": {
            name: {f"slot_{index}": words for index, words in enumerate(prompt_words)}
            for name, prompt_words in WAVES
        },
    }
    return value


MODE_CONFIGS = {mode: mode_config(mode) for mode in MODES}
mixed.VARIANT_CONFIGS.update(MODE_CONFIGS)
EFFECTIVE_CONFIG = {
    "build": MODE_CONFIGS[MODES[0]]["build"],
    "comparison": MODE_CONFIGS[MODES[0]]["workload"],
    "legacy_host_barriers": {
        "runtime": MODE_CONFIGS["legacy_host_barriers"]["runtime"],
        "server": MODE_CONFIGS["legacy_host_barriers"]["server"],
    },
    "stream_ordered": {
        "runtime": MODE_CONFIGS["stream_ordered"]["runtime"],
        "server": MODE_CONFIGS["stream_ordered"]["server"],
    },
}


def _mode_metrics(prefix: str) -> dict[str, tuple[str, str, bool]]:
    return {
        f"{prefix}_completion_token_count": ("tokens", "sum", False),
        f"{prefix}_device_fault_count": ("count", "sum", True),
        f"{prefix}_device_wait_count": ("count", "sum", True),
        f"{prefix}_duration_seconds": ("s", "sum", True),
        f"{prefix}_e2e_ms_max": ("ms", "max", True),
        f"{prefix}_e2e_ms_p50": ("ms", "p50", True),
        f"{prefix}_e2e_ms_p99": ("ms", "p99", True),
        f"{prefix}_external_yield_device_wait_count": ("count", "sum", True),
        f"{prefix}_external_yield_stream_wait_count": ("count", "sum", True),
        f"{prefix}_graph_activity_count": ("count", "sum", True),
        f"{prefix}_gpu_memory_peak_bytes": ("bytes", "max", True),
        f"{prefix}_itl_ms_max": ("ms", "max", True),
        f"{prefix}_itl_ms_p50": ("ms", "p50", True),
        f"{prefix}_itl_ms_p99": ("ms", "p99", True),
        f"{prefix}_memory_sample_count": ("count", "sum", False),
        f"{prefix}_memory_sampler_error_count": ("count", "sum", True),
        f"{prefix}_mutation_event_count": ("count", "sum", True),
        f"{prefix}_output_token_throughput_per_second": ("tokens/s", "mean", False),
        f"{prefix}_pause_count": ("count", "sum", True),
        f"{prefix}_request_count": ("count", "sum", False),
        f"{prefix}_request_failure_count": ("count", "sum", True),
        f"{prefix}_skipped_barrier_count": ("count", "sum", False),
        f"{prefix}_stall_count": ("count", "sum", True),
        f"{prefix}_stream_wait_count": ("count", "sum", True),
        f"{prefix}_synchronization_wait_ms": ("ms", "sum", True),
        f"{prefix}_ttft_ms_max": ("ms", "max", True),
        f"{prefix}_ttft_ms_p50": ("ms", "p50", True),
        f"{prefix}_ttft_ms_p99": ("ms", "p99", True),
    }


METRIC_DEFINITIONS = {
    **_mode_metrics("legacy"),
    **_mode_metrics("stream_ordered"),
    "correctness_action_token_count": ("tokens", "exact", False),
    "correctness_behavior_logprob_mismatch_count": ("count", "sum", True),
    "correctness_non_finite_logprob_count": ("count", "sum", True),
    "correctness_output_mismatch_count": ("count", "sum", True),
    "correctness_token_id_mismatch_count": ("count", "sum", True),
    "execution_failure_count": ("count", "sum", True),
    "output_contract_mismatch_count": ("count", "sum", True),
    "policy_attestation_failure_count": ("count", "sum", True),
    "prometheus_attestation_failure_count": ("count", "sum", True),
    "shutdown_forced_count": ("count", "sum", True),
    "shutdown_nonzero_count": ("count", "sum", True),
    "snapshot_residue_count": ("count", "sum", True),
}


@dataclasses.dataclass(frozen=True)
class ReasonStats:
    device_wait_count: int
    stream_wait_count: int
    waited_ns: int
    skipped_count: int


@dataclasses.dataclass(frozen=True)
class SyncSnapshot:
    mode: str
    reasons: dict[str, ReasonStats]

    @property
    def device_wait_count(self) -> int:
        return sum(value.device_wait_count for value in self.reasons.values())

    @property
    def stream_wait_count(self) -> int:
        return sum(value.stream_wait_count for value in self.reasons.values())

    @property
    def waited_ns(self) -> int:
        return sum(value.waited_ns for value in self.reasons.values())

    @property
    def skipped_count(self) -> int:
        return sum(value.skipped_count for value in self.reasons.values())


@dataclasses.dataclass(frozen=True)
class ArmRun:
    mode: str
    results: tuple[mixed.StreamResult, ...]
    correctness_record: correctness.CompletionRecord
    elapsed_seconds: float
    sync_delta: SyncSnapshot
    peak_memory_bytes: int
    memory_sample_count: int
    memory_sampler_error_count: int
    policy_failures: tuple[str, ...]
    prometheus_failures: tuple[str, ...]
    device_fault_count: int
    graph_activity_count: int
    mutation_event_count: int
    shutdown: mixed.ShutdownOutcome
    snapshot_residue: tuple[str, ...]


def require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SynchronizationQualificationError(
            f"{label} must be a nonnegative integer, got {value!r}"
        )
    return value


def expected_policy(mode: str) -> dict[str, Any]:
    return {
        "schema_id": "kiln.accelerator-runtime-policy.v2",
        "version": 2,
        "serving_profile": "experimental",
        "serving_profile_source": "config_file",
        "rocm_synchronization_mode": {
            "configured": mode,
            "effective": mode,
            "source": "config_file",
        },
        "rocm_graph_mode": {
            "configured": "disabled",
            "effective": "disabled",
            "source": "config_file",
        },
        "rocm_graph_cache_entries": {
            "configured": 8,
            "effective": 8,
            "source": "config_file",
        },
        "rocm_graph_cache_max_bytes": {
            "configured": 1 << 30,
            "effective": 1 << 30,
            "source": "config_file",
        },
    }


def policy_attestation_failures(value: Any, mode: str, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} accelerator policy is missing"]
    expected = expected_policy(mode)
    return [
        f"{label}.{field}={value.get(field)!r}, expected {expected_value!r}"
        for field, expected_value in expected.items()
        if value.get(field) != expected_value
    ]


def synchronization_snapshot(health: Any, mode: str) -> SyncSnapshot:
    if not isinstance(health, dict):
        raise SynchronizationQualificationError("health response is not an object")
    runtime = health.get("decode_runtime")
    if not isinstance(runtime, dict):
        raise SynchronizationQualificationError("health.decode_runtime is missing")
    stats = runtime.get("rocm_synchronization")
    if not isinstance(stats, dict):
        raise SynchronizationQualificationError(
            "health.decode_runtime.rocm_synchronization is missing"
        )
    if stats.get("active") is not True or stats.get("telemetry_available") is not True:
        raise SynchronizationQualificationError(
            "ROCm synchronization telemetry is not active and available"
        )
    if stats.get("telemetry_error") is not None:
        raise SynchronizationQualificationError(
            f"ROCm synchronization telemetry reported {stats.get('telemetry_error')!r}"
        )
    if stats.get("cleanup_quarantined") is not False:
        raise SynchronizationQualificationError(
            "ROCm cleanup quarantine is set or missing"
        )
    raw_reasons = stats.get("reasons")
    if not isinstance(raw_reasons, list):
        raise SynchronizationQualificationError("ROCm reason telemetry is missing")
    observed_names = tuple(
        item.get("reason") if isinstance(item, dict) else None for item in raw_reasons
    )
    if observed_names != ROCM_SYNC_REASONS:
        raise SynchronizationQualificationError(
            f"ROCm reason dimensions drifted: {observed_names!r}"
        )
    reasons: dict[str, ReasonStats] = {}
    for item in raw_reasons:
        assert isinstance(item, dict)
        reason = item["reason"]
        reasons[reason] = ReasonStats(
            device_wait_count=require_nonnegative_int(
                item.get("device_wait_count"), f"{reason}.device_wait_count"
            ),
            stream_wait_count=require_nonnegative_int(
                item.get("stream_wait_count"), f"{reason}.stream_wait_count"
            ),
            waited_ns=require_nonnegative_int(item.get("waited_ns"), f"{reason}.waited_ns"),
            skipped_count=require_nonnegative_int(
                item.get("skipped_count"), f"{reason}.skipped_count"
            ),
        )
    snapshot = SyncSnapshot(mode=mode, reasons=reasons)
    reported_totals = {
        "total_device_wait_count": snapshot.device_wait_count,
        "total_stream_wait_count": snapshot.stream_wait_count,
        "total_waited_ns": snapshot.waited_ns,
        "total_skipped_count": snapshot.skipped_count,
    }
    for field, expected in reported_totals.items():
        if require_nonnegative_int(stats.get(field), field) != expected:
            raise SynchronizationQualificationError(
                f"health {field} does not equal the reason sum"
            )
    return snapshot


def synchronization_delta(before: SyncSnapshot, after: SyncSnapshot) -> SyncSnapshot:
    if before.mode != after.mode or tuple(before.reasons) != tuple(after.reasons):
        raise SynchronizationQualificationError("synchronization snapshot identity drifted")
    reasons: dict[str, ReasonStats] = {}
    for reason in before.reasons:
        left = before.reasons[reason]
        right = after.reasons[reason]
        fields = (
            right.device_wait_count - left.device_wait_count,
            right.stream_wait_count - left.stream_wait_count,
            right.waited_ns - left.waited_ns,
            right.skipped_count - left.skipped_count,
        )
        if any(value < 0 for value in fields):
            raise SynchronizationQualificationError(
                f"synchronization counters regressed for {reason}"
            )
        reasons[reason] = ReasonStats(*fields)
    return SyncSnapshot(mode=before.mode, reasons=reasons)


POLICY_METRIC_RE = re.compile(
    r'^kiln_rocm_synchronization_policy_info\{mode="([a-z_]+)"\} ([0-9.eE+-]+)$'
)
REASON_METRIC_RE = re.compile(
    r'^kiln_rocm_synchronizations_total\{reason="([a-z_]+)",scope="(device|stream)"\} ([0-9.eE+-]+)$'
)
WAIT_METRIC_RE = re.compile(
    r'^kiln_rocm_synchronization_wait_seconds_total\{reason="([a-z_]+)"\} ([0-9.eE+-]+)$'
)
SKIP_METRIC_RE = re.compile(
    r'^kiln_rocm_synchronization_skipped_total\{reason="([a-z_]+)"\} ([0-9.eE+-]+)$'
)
QUARANTINE_METRIC_RE = re.compile(
    r"^kiln_rocm_cleanup_quarantined ([0-9.eE+-]+)$"
)


def prometheus_sync_snapshot(text: str, mode: str) -> SyncSnapshot:
    policy_modes: list[str] = []
    quarantine_values: list[float] = []
    values: dict[str, dict[str, int | float]] = {}
    for line in text.splitlines():
        if match := POLICY_METRIC_RE.fullmatch(line):
            if float(match.group(2)) == 1.0:
                policy_modes.append(match.group(1))
            continue
        if match := QUARANTINE_METRIC_RE.fullmatch(line):
            quarantine_values.append(float(match.group(1)))
            continue
        for pattern, field in (
            (REASON_METRIC_RE, None),
            (WAIT_METRIC_RE, "waited_seconds"),
            (SKIP_METRIC_RE, "skipped_count"),
        ):
            match = pattern.fullmatch(line)
            if match is None:
                continue
            reason = match.group(1)
            reason_values = values.setdefault(reason, {})
            if field is None:
                field = f"{match.group(2)}_wait_count"
                raw = match.group(3)
            else:
                raw = match.group(2)
            if field in reason_values:
                raise SynchronizationQualificationError(
                    f"Prometheus repeated {reason}.{field}"
                )
            if field == "waited_seconds":
                number: int | float = float(raw)
                if not math.isfinite(number) or number < 0:
                    raise SynchronizationQualificationError(
                        f"Prometheus {reason}.{field} is invalid"
                    )
            else:
                if re.fullmatch(r"[0-9]+", raw) is None:
                    raise SynchronizationQualificationError(
                        f"Prometheus {reason}.{field} is not an exact integer"
                    )
                number = int(raw)
            reason_values[field] = number
            break
    if policy_modes != [mode]:
        raise SynchronizationQualificationError(
            f"Prometheus policy modes={policy_modes!r}, expected {[mode]!r}"
        )
    if quarantine_values != [0.0]:
        raise SynchronizationQualificationError(
            "Prometheus ROCm cleanup quarantine must be present exactly once and equal zero; "
            f"got {quarantine_values!r}"
        )
    if tuple(values) != ROCM_SYNC_REASONS:
        raise SynchronizationQualificationError(
            f"Prometheus reason dimensions drifted: {tuple(values)!r}"
        )
    reasons = {
        reason: ReasonStats(
            device_wait_count=_exact_metric_integer(
                fields.get("device_wait_count"), f"{reason}.device_wait_count"
            ),
            stream_wait_count=_exact_metric_integer(
                fields.get("stream_wait_count"), f"{reason}.stream_wait_count"
            ),
            waited_ns=round(
                _required_metric(fields.get("waited_seconds"), f"{reason}.waited_seconds")
                * 1_000_000_000.0
            ),
            skipped_count=_exact_metric_integer(
                fields.get("skipped_count"), f"{reason}.skipped_count"
            ),
        )
        for reason, fields in values.items()
    }
    return SyncSnapshot(mode=mode, reasons=reasons)


def _required_metric(value: int | float | None, label: str) -> float:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SynchronizationQualificationError(f"Prometheus omitted {label}")
    return float(value)


def _exact_metric_integer(value: int | float | None, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SynchronizationQualificationError(f"Prometheus {label} is not integral")
    return value


def prometheus_attestation_failures(
    health: SyncSnapshot, prometheus: SyncSnapshot
) -> list[str]:
    failures: list[str] = []
    if health.mode != prometheus.mode:
        failures.append("health and Prometheus synchronization modes differ")
    for reason in ROCM_SYNC_REASONS:
        left = health.reasons[reason]
        right = prometheus.reasons[reason]
        if left.device_wait_count != right.device_wait_count:
            failures.append(f"{reason} device-wait count differs")
        if left.stream_wait_count != right.stream_wait_count:
            failures.append(f"{reason} stream-wait count differs")
        if abs(left.waited_ns - right.waited_ns) > 1:
            failures.append(f"{reason} waited duration differs")
        if left.skipped_count != right.skipped_count:
            failures.append(f"{reason} skipped count differs")
    return failures


def run_wave(
    port: int,
    *,
    mode: str,
    wave_name: str,
    prompt_words: tuple[int, ...],
    base_seed: int,
    deadline: float,
) -> list[mixed.StreamResult]:
    abort = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(prompt_words))
    futures: list[concurrent.futures.Future[mixed.StreamResult]] = []
    try:
        for index, words in enumerate(prompt_words):
            role = f"{wave_name}-{index:02d}"
            futures.append(
                pool.submit(
                    mixed.run_stream,
                    port,
                    name=role,
                    marker=mixed.workload_marker(base_seed, role),
                    prompt_words=words,
                    max_tokens=MAX_TOKENS,
                    seed=base_seed + index,
                    absolute_deadline=min(deadline, time.monotonic() + REQUEST_TIMEOUT_SECONDS),
                    abort_event=abort,
                )
            )
        return [
            future.result(timeout=mixed.remaining_until(deadline, f"{mode} {wave_name}"))
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
            raise SynchronizationQualificationError(
                f"{mode} {wave_name} left {len(unfinished)} request workers"
            )


def run_arm(
    binary: Path, model_path: Path, seed: int, mode: str, deadline: float
) -> ArmRun:
    policy_events_started = time.monotonic()
    port = mixed.free_loopback_port()
    run_dir = mixed.create_serving_run_dir(f"sync-ab-{mode}")
    adapter_dir = run_dir / "adapters"
    snapshot_dir = run_dir / "model-snapshots"
    config_path = run_dir / "kiln.toml"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    mixed.write_server_config(
        config_path,
        mode,
        model_path,
        port,
        adapter_dir,
        snapshot_dir,
        rocm_synchronization_mode=mode,
        rocm_graph_mode="disabled",
    )
    process, server_log = mixed.start_server(binary, config_path, mode)
    sampler = mixed.MemorySampler(port)
    shutdown: mixed.ShutdownOutcome | None = None
    residue: tuple[str, ...] = ()
    result: ArmRun | None = None
    measured_started = time.monotonic()
    try:
        mixed.wait_ready(port, process, server_log, deadline)
        startup_health = mixed.read_stable_health(port, deadline, f"{mode} startup")
        startup_config = mixed.json_request(port, "GET", "/v1/config")
        policy_failures = [
            *policy_attestation_failures(
                (startup_health.get("decode_runtime") or {}).get("accelerator_runtime"),
                mode,
                "health.decode_runtime.accelerator_runtime",
            ),
            *policy_attestation_failures(
                startup_config.get("accelerator_runtime"),
                mode,
                "config.accelerator_runtime",
            ),
        ]
        startup_snapshot = synchronization_snapshot(startup_health, mode)
        before_prometheus = prometheus_sync_snapshot(
            mixed.text_request(port, "/metrics"), mode
        )
        prometheus_failures = prometheus_attestation_failures(
            startup_snapshot, before_prometheus
        )
        warmup = mixed.run_stream(
            port,
            name="warmup",
            marker=mixed.workload_marker(seed, "sync-ab-warmup"),
            prompt_words=16,
            max_tokens=WARMUP_MAX_TOKENS,
            seed=seed,
            absolute_deadline=min(deadline, time.monotonic() + REQUEST_TIMEOUT_SECONDS),
        )
        if not warmup.success or warmup.completion_tokens != WARMUP_MAX_TOKENS:
            raise SynchronizationQualificationError(
                f"{mode} warmup failed its fixed-output contract: {warmup}"
            )
        correctness_record = correctness.completion_request(
            port,
            "synchronization-policy",
            mixed.deterministic_prompt("QUAL-SYNC-POLICY", 64),
            seed + 10_000,
            max_tokens=16,
            timeout_seconds=mixed.remaining_until(
                deadline, f"{mode} correctness probe", REQUEST_TIMEOUT_SECONDS
            ),
        )
        measurement_health = mixed.read_stable_health(
            port, deadline, f"{mode} post-warmup telemetry"
        )
        before = synchronization_snapshot(measurement_health, mode)
        measurement_prometheus = prometheus_sync_snapshot(
            mixed.text_request(port, "/metrics"), mode
        )
        prometheus_failures.extend(
            prometheus_attestation_failures(before, measurement_prometheus)
        )
        policy_failures.extend(
            policy_attestation_failures(
                (measurement_health.get("decode_runtime") or {}).get(
                    "accelerator_runtime"
                ),
                mode,
                "post-warmup health.decode_runtime.accelerator_runtime",
            )
        )
        sampler.start()
        measured_started = time.monotonic()
        results: list[mixed.StreamResult] = []
        wave_cursor = before
        for wave_index, (wave_name, words) in enumerate(WAVES):
            wave_results = run_wave(
                port,
                mode=mode,
                wave_name=wave_name,
                prompt_words=words,
                base_seed=seed + 100 + wave_index * 100,
                deadline=deadline,
            )
            results.extend(wave_results)
            wave_health = mixed.read_stable_health(
                port, deadline, f"{mode} {wave_name} telemetry"
            )
            wave_snapshot = synchronization_snapshot(wave_health, mode)
            wave_prometheus = prometheus_sync_snapshot(
                mixed.text_request(port, "/metrics"), mode
            )
            prometheus_failures.extend(
                prometheus_attestation_failures(wave_snapshot, wave_prometheus)
            )
            policy_failures.extend(
                policy_attestation_failures(
                    (wave_health.get("decode_runtime") or {}).get(
                        "accelerator_runtime"
                    ),
                    mode,
                    f"{wave_name} health.decode_runtime.accelerator_runtime",
                )
            )
            wave_delta = synchronization_delta(wave_cursor, wave_snapshot)
            wave_itl = [gap for result in wave_results for gap in result.itl_ms]
            mixed.trace(
                "rocm_sync_ab_wave",
                itl_max_ms=max(wave_itl, default=0.0),
                mode=mode,
                pause_count=sum(gap >= PAUSE_GATE_MS for gap in wave_itl),
                reasons={
                    reason: dataclasses.asdict(stats)
                    for reason, stats in wave_delta.reasons.items()
                },
                stall_count=sum(gap >= STALL_THRESHOLD_MS for gap in wave_itl),
                wave=wave_name,
            )
            wave_cursor = wave_snapshot
        elapsed_seconds = time.monotonic() - measured_started
        sampler.close()
        final_health = mixed.read_stable_health(port, deadline, f"{mode} final")
        after = synchronization_snapshot(final_health, mode)
        after_prometheus = prometheus_sync_snapshot(
            mixed.text_request(port, "/metrics"), mode
        )
        prometheus_failures.extend(
            prometheus_attestation_failures(after, after_prometheus)
        )
        policy_failures.extend(
            policy_attestation_failures(
                (final_health.get("decode_runtime") or {}).get("accelerator_runtime"),
                mode,
                "final health.decode_runtime.accelerator_runtime",
            )
        )
        sync_delta = synchronization_delta(before, after)
        events = server_log.events_since(policy_events_started)
        graph = mixed.graph_snapshot(final_health)
        graph_activity_count = sum(
            graph[field]
            for field in (
                "capture_attempts",
                "capture_successes",
                "capture_failures",
                "replay_attempts",
                "replay_successes",
                "replay_failures",
                "captured_graph_count",
                "graph_slot_create_count",
                "graph_slot_reuse_count",
                "fallback_total",
            )
        )
        result = ArmRun(
            mode=mode,
            results=tuple(results),
            correctness_record=correctness_record,
            elapsed_seconds=elapsed_seconds,
            sync_delta=sync_delta,
            peak_memory_bytes=max(sampler.samples, default=0),
            memory_sample_count=len(sampler.samples),
            memory_sampler_error_count=len(sampler.errors),
            policy_failures=tuple(policy_failures),
            prometheus_failures=tuple(prometheus_failures),
            device_fault_count=sum(event.category == "device_fault" for event in events),
            graph_activity_count=graph_activity_count,
            mutation_event_count=sum(
                event.category
                in {
                    "graph_capture",
                    "graph_fallback",
                    "graph_sync",
                    "kv_resize",
                    "memory_reclaim",
                }
                for event in events
            ),
            shutdown=mixed.ShutdownOutcome(0, False, 0.0),
            snapshot_residue=(),
        )
        mixed.trace(
            "rocm_sync_ab_arm",
            mode=mode,
            reasons={
                reason: dataclasses.asdict(stats)
                for reason, stats in sync_delta.reasons.items()
            },
            request_count=len(results),
        )
    finally:
        sampler.close()
        shutdown = mixed.terminate_process(process)
        server_log.join()
        residue = tuple(mixed.snapshot_payload_residue(snapshot_dir))
        shutil.rmtree(run_dir, ignore_errors=True)
    if result is None or shutdown is None:
        raise AssertionError(f"{mode} arm completed without a result")
    return dataclasses.replace(
        result, shutdown=shutdown, snapshot_residue=residue
    )


def _prefix(mode: str) -> str:
    return "legacy" if mode == "legacy_host_barriers" else "stream_ordered"


def arm_metric_values(arm: ArmRun) -> dict[str, int | float]:
    prefix = _prefix(arm.mode)
    successes = [result for result in arm.results if result.success]
    ttft = [result.ttft_ms for result in successes]
    e2e = [result.e2e_ms for result in successes]
    itl = [gap for result in successes for gap in result.itl_ms]
    external = arm.sync_delta.reasons["external_yield"]
    completion_tokens = sum(result.completion_tokens for result in successes)
    return {
        f"{prefix}_completion_token_count": completion_tokens,
        f"{prefix}_device_fault_count": arm.device_fault_count,
        f"{prefix}_device_wait_count": arm.sync_delta.device_wait_count,
        f"{prefix}_duration_seconds": arm.elapsed_seconds,
        f"{prefix}_e2e_ms_max": max(e2e, default=0.0),
        f"{prefix}_e2e_ms_p50": mixed.percentile_r7(e2e, 0.5),
        f"{prefix}_e2e_ms_p99": mixed.percentile_r7(e2e, 0.99),
        f"{prefix}_external_yield_device_wait_count": external.device_wait_count,
        f"{prefix}_external_yield_stream_wait_count": external.stream_wait_count,
        f"{prefix}_graph_activity_count": arm.graph_activity_count,
        f"{prefix}_gpu_memory_peak_bytes": arm.peak_memory_bytes,
        f"{prefix}_itl_ms_max": max(itl, default=0.0),
        f"{prefix}_itl_ms_p50": mixed.percentile_r7(itl, 0.5),
        f"{prefix}_itl_ms_p99": mixed.percentile_r7(itl, 0.99),
        f"{prefix}_memory_sample_count": arm.memory_sample_count,
        f"{prefix}_memory_sampler_error_count": arm.memory_sampler_error_count,
        f"{prefix}_mutation_event_count": arm.mutation_event_count,
        f"{prefix}_output_token_throughput_per_second": (
            completion_tokens / max(arm.elapsed_seconds, 1e-9)
        ),
        f"{prefix}_pause_count": sum(gap >= PAUSE_GATE_MS for gap in itl),
        f"{prefix}_request_count": len(arm.results),
        f"{prefix}_request_failure_count": len(arm.results) - len(successes),
        f"{prefix}_skipped_barrier_count": arm.sync_delta.skipped_count,
        f"{prefix}_stall_count": sum(gap >= STALL_THRESHOLD_MS for gap in itl),
        f"{prefix}_stream_wait_count": arm.sync_delta.stream_wait_count,
        f"{prefix}_synchronization_wait_ms": arm.sync_delta.waited_ns / 1_000_000.0,
        f"{prefix}_ttft_ms_max": max(ttft, default=0.0),
        f"{prefix}_ttft_ms_p50": mixed.percentile_r7(ttft, 0.5),
        f"{prefix}_ttft_ms_p99": mixed.percentile_r7(ttft, 0.99),
    }


def output_contract_mismatches(
    legacy: ArmRun, stream_ordered: ArmRun
) -> list[str]:
    failures: list[str] = []
    if len(legacy.results) != sum(len(words) for _, words in WAVES):
        failures.append("legacy arm request count drifted")
    if len(stream_ordered.results) != len(legacy.results):
        failures.append("A/B request counts differ")
    for left, right in zip(legacy.results, stream_ordered.results, strict=False):
        for arm, result in ((legacy.mode, left), (stream_ordered.mode, right)):
            if not result.success:
                failures.append(f"{arm} {result.name} failed: {result.error}")
            if result.finish_reason != "length" or result.completion_tokens != MAX_TOKENS:
                failures.append(
                    f"{arm} {result.name} violated fixed output: "
                    f"{result.finish_reason!r}/{result.completion_tokens}"
                )
        if left.name != right.name:
            failures.append(f"A/B request identity differs: {left.name!r}/{right.name!r}")
        if left.prompt_tokens != right.prompt_tokens:
            failures.append(f"{left.name} prompt token count differs across A/B")
        if left.completion_tokens != right.completion_tokens:
            failures.append(f"{left.name} completion token count differs across A/B")
    return failures


def correctness_metric_values(
    legacy: ArmRun, stream_ordered: ArmRun
) -> dict[str, int]:
    mismatches = correctness.mismatch_counts(
        (legacy.correctness_record,),
        (stream_ordered.correctness_record,),
    )
    return {
        "correctness_action_token_count": len(legacy.correctness_record.action_tokens),
        "correctness_behavior_logprob_mismatch_count": mismatches[
            "behavior_logprob_mismatch_count"
        ],
        "correctness_non_finite_logprob_count": sum(
            not math.isfinite(logprob)
            for arm in (legacy, stream_ordered)
            for logprob in arm.correctness_record.sampled_logprobs
        ),
        "correctness_output_mismatch_count": mismatches["output_mismatch_count"],
        "correctness_token_id_mismatch_count": mismatches["token_id_mismatch_count"],
    }


def metrics_from_values(values: dict[str, int | float]) -> list[dict[str, Any]]:
    if set(values) != set(METRIC_DEFINITIONS):
        missing = sorted(set(METRIC_DEFINITIONS) - set(values))
        extra = sorted(set(values) - set(METRIC_DEFINITIONS))
        raise SynchronizationQualificationError(
            f"metric set mismatch: missing={missing}, extra={extra}"
        )
    metrics: list[dict[str, Any]] = []
    for name, (unit, aggregation, lower_is_better) in sorted(
        METRIC_DEFINITIONS.items()
    ):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise SynchronizationQualificationError(
                f"metric {name} must be finite and nonnegative"
            )
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


def execute(model_path: Path, seed: int) -> tuple[list[dict[str, Any]], str | None]:
    deadline = time.monotonic() + OVERALL_TIMEOUT_SECONDS
    values: dict[str, int | float] = {name: 0 for name in METRIC_DEFINITIONS}
    failures: list[str] = []
    arms: dict[str, ArmRun] = {}
    try:
        binary, binary_hash, build_seconds = mixed.build_binary(deadline)
        mixed.trace(
            "rocm_sync_ab_binary_built",
            build_seconds=build_seconds,
            path=str(binary.relative_to(ROOT)),
            sha256=binary_hash,
        )
        for mode in MODES:
            arms[mode] = run_arm(binary, model_path, seed, mode, deadline)
            values.update(arm_metric_values(arms[mode]))
        legacy = arms["legacy_host_barriers"]
        stream_ordered = arms["stream_ordered"]
        values.update(correctness_metric_values(legacy, stream_ordered))
        output_failures = output_contract_mismatches(legacy, stream_ordered)
        policy_failures = [
            failure for arm in arms.values() for failure in arm.policy_failures
        ]
        prometheus_failures = [
            failure for arm in arms.values() for failure in arm.prometheus_failures
        ]
        values["output_contract_mismatch_count"] = len(output_failures)
        values["policy_attestation_failure_count"] = len(policy_failures)
        values["prometheus_attestation_failure_count"] = len(prometheus_failures)
        values["shutdown_forced_count"] = sum(
            arm.shutdown.forced for arm in arms.values()
        )
        values["shutdown_nonzero_count"] = sum(
            arm.shutdown.returncode != 0 for arm in arms.values()
        )
        values["snapshot_residue_count"] = sum(
            len(arm.snapshot_residue) for arm in arms.values()
        )
        failures.extend(output_failures)
        failures.extend(policy_failures)
        failures.extend(prometheus_failures)
        if values["correctness_action_token_count"] < 1:
            failures.append("correctness probe returned no action tokens")
        for arm in arms.values():
            prefix = _prefix(arm.mode)
            if arm.device_fault_count:
                failures.append(f"{arm.mode} recorded a device fault")
            if arm.graph_activity_count:
                failures.append(f"{arm.mode} recorded graph activity while disabled")
            if arm.mutation_event_count:
                failures.append(
                    f"{arm.mode} recorded a graph, resize, reclaim, or trim event"
                )
            if arm.memory_sample_count < 1:
                failures.append(f"{arm.mode} collected no GPU memory samples")
            if arm.memory_sampler_error_count:
                failures.append(f"{arm.mode} GPU memory sampling failed")
            if values[f"{prefix}_pause_count"]:
                failures.append(
                    f"{arm.mode} recorded {values[f'{prefix}_pause_count']} "
                    f"ITL gaps >= {PAUSE_GATE_MS:.0f} ms"
                )
        if legacy.sync_delta.reasons["external_yield"].device_wait_count < 1:
            failures.append("legacy arm exercised no external-yield device wait")
        if stream_ordered.sync_delta.reasons["external_yield"].stream_wait_count < 1:
            failures.append("stream-ordered arm exercised no external-yield stream wait")
        if stream_ordered.sync_delta.skipped_count < 1:
            failures.append("stream-ordered arm skipped no proven same-stream barrier")
        if legacy.sync_delta.skipped_count != 0:
            failures.append("legacy arm unexpectedly skipped a synchronization barrier")
        for metric in (
            "output_contract_mismatch_count",
            "policy_attestation_failure_count",
            "prometheus_attestation_failure_count",
            "shutdown_forced_count",
            "shutdown_nonzero_count",
            "snapshot_residue_count",
            "execution_failure_count",
            "correctness_behavior_logprob_mismatch_count",
            "correctness_non_finite_logprob_count",
            "correctness_output_mismatch_count",
            "correctness_token_id_mismatch_count",
        ):
            if values[metric] != 0:
                failures.append(f"{metric}={values[metric]}, expected 0")
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        values["execution_failure_count"] = 1

    details: str | None
    if failures:
        details = mixed.bounded_details(" | ".join(dict.fromkeys(failures)))
    else:
        details = json.dumps(
            {
                "legacy_stalls": values["legacy_stall_count"],
                "legacy_sync_wait_ms": values["legacy_synchronization_wait_ms"],
                "legacy_trace_sha256": correctness.canonical_hash(
                    (arms["legacy_host_barriers"].correctness_record,)
                ),
                "stream_ordered_skipped_barriers": values[
                    "stream_ordered_skipped_barrier_count"
                ],
                "stream_ordered_stalls": values["stream_ordered_stall_count"],
                "stream_ordered_sync_wait_ms": values[
                    "stream_ordered_synchronization_wait_ms"
                ],
                "stream_ordered_trace_sha256": correctness.canonical_hash(
                    (arms["stream_ordered"].correctness_record,)
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    return metrics_from_values(values), details


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not args.model_path.is_dir():
        parser.error("--model-path must be a directory")
    return args


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    variant = os.environ.get(VARIANT_ENV, "")
    result_value = os.environ.get(RESULT_ENV)
    if variant != VARIANT_ID:
        print(
            f"{VARIANT_ENV} must be {VARIANT_ID!r}, got {variant!r}",
            file=os.sys.stderr,
        )
        return 2
    if not result_value:
        print(f"{RESULT_ENV} is required", file=os.sys.stderr)
        return 2
    metrics, details = execute(args.model_path.resolve(), args.seed)
    passed = details is not None and details.startswith("{")
    result = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "status": "passed" if passed else "failed",
        "duration_seconds": time.monotonic() - started,
        "effective_config": EFFECTIVE_CONFIG,
        "metrics": metrics,
        "tolerances": [],
        "details": details,
    }
    mixed.write_result(Path(result_value), result)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
