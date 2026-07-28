#!/usr/bin/env python3
"""Run the bounded ROCm/Vulkan serving regression-closure oracle."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import serve_development_soak as soak
import serve_mixed_load as mixed


ROOT = Path(__file__).resolve().parents[2]
RESULT_ENV = mixed.RESULT_ENV
VARIANT_ENV = mixed.VARIANT_ENV
ROCM_VARIANT = "rocm-kv-prefix-closure"
VULKAN_VARIANT = "vulkan-kv-prefix-quarantine"
VARIANTS = (ROCM_VARIANT, VULKAN_VARIANT)
CASE_IDS = {
    ROCM_VARIANT: "rocm-serving-regression-closure",
    VULKAN_VARIANT: "vulkan-serving-regression-closure",
}
BACKENDS = {ROCM_VARIANT: "rocm", VULKAN_VARIANT: "vulkan"}
DEVICES = {ROCM_VARIANT: "rocm:0", VULKAN_VARIANT: "vulkan:0"}
BUILD_SPECS = {
    ROCM_VARIANT: mixed.ROCM_BUILD_SPEC,
    VULKAN_VARIANT: mixed.VULKAN_BUILD_SPEC,
}
BUILD_TIMEOUT_SECONDS = 900.0
OVERALL_TIMEOUT_SECONDS = 1800.0
REQUEST_TIMEOUT_SECONDS = 600.0
KV_BLOCK_SIZE = 16
KV_NUM_BLOCKS = 32
PREFIX_CACHE_MAX_BLOCKS = 16
PREFIX_CACHE_MAX_ENTRIES = 4
PROMPT_WORDS = 48
PRIME_MAX_TOKENS = 16
PRESSURE_MAX_TOKENS = 256
EXPECTED_REQUESTS = 2
EXPECTED_COMPLETION_TOKENS = PRIME_MAX_TOKENS + PRESSURE_MAX_TOKENS


def effective_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported closure variant {variant!r}")
    config = mixed._variant_config(
        serving_profile="experimental",
        kv_autoscale_requested=False,
        kv_autoscale_enabled=False,
        memory_reclaim_requested_mode="off",
        memory_reclaim_mode="off",
        rocm_graphs_requested=False,
        rocm_graphs_enabled=False,
        request_timeout_seconds=int(REQUEST_TIMEOUT_SECONDS),
        max_decode_batch=1,
        max_prefill_layers_per_cycle=4,
    )
    spec = BUILD_SPECS[variant]
    config["build"] = {
        **spec.effective_config(),
        "timeout_seconds": int(BUILD_TIMEOUT_SECONDS),
    }
    if variant == VULKAN_VARIANT:
        config["model"].update(
            {
                "vulkan_decode_weight_prewarm": (
                    mixed.VULKAN_DECODE_WEIGHT_PREWARM
                ),
                "vulkan_decode_weight_prewarm_mib_per_second": (
                    mixed.VULKAN_DECODE_WEIGHT_PREWARM_MIB_PER_SECOND
                ),
            }
        )
    prefix_enabled = variant == ROCM_VARIANT
    config["runtime"].update(
        {
            "prefix_cache_requested_enabled": True,
            "prefix_cache_effective_enabled": prefix_enabled,
            "prefix_cache_effective_reason": (
                "active" if prefix_enabled else "vulkan_correctness_quarantine"
            ),
        }
    )
    config["memory"] = {
        "num_blocks": KV_NUM_BLOCKS,
        "kv_autoscale": False,
        "reclaim_mode": "off",
    }
    config["prefix_cache"] = {
        "requested_enabled": True,
        "effective_enabled": prefix_enabled,
        "max_blocks": PREFIX_CACHE_MAX_BLOCKS,
        "max_entries": PREFIX_CACHE_MAX_ENTRIES,
    }
    config["workload"] = {
        "expected_completion_tokens": EXPECTED_COMPLETION_TOKENS,
        "expected_finish_reason": "length",
        "expected_request_count": EXPECTED_REQUESTS,
        "kv_block_size": KV_BLOCK_SIZE,
        "kv_num_blocks": KV_NUM_BLOCKS,
        "overall_timeout_seconds": int(OVERALL_TIMEOUT_SECONDS),
        "prefix_cache_max_blocks": PREFIX_CACHE_MAX_BLOCKS,
        "prefix_cache_max_entries": PREFIX_CACHE_MAX_ENTRIES,
        "pressure_max_tokens": PRESSURE_MAX_TOKENS,
        "prime_max_tokens": PRIME_MAX_TOKENS,
        "prompt_words": PROMPT_WORDS,
        "request_timeout_seconds": int(REQUEST_TIMEOUT_SECONDS),
        "response_oracle": mixed.RESPONSE_ORACLE,
        "sequence": "prime_then_pressure",
    }
    return config


EFFECTIVE_CONFIGS = {variant: effective_config(variant) for variant in VARIANTS}
mixed.VARIANT_CONFIGS.update(EFFECTIVE_CONFIGS)


METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    "batching_error_count": ("count", "sum", True),
    "binary_build_count": ("count", "sum", True),
    "completion_token_count": ("tokens", "sum", False),
    "device_fault_event_count": ("count", "sum", True),
    "exact_output_failure_count": ("count", "sum", True),
    "external_yield_sync_call_count": ("count", "sum", False),
    "external_yield_sync_failure_count": ("count", "sum", True),
    "external_yield_sync_max_ms": ("ms", "max", True),
    "external_yield_sync_slow_count": ("count", "sum", True),
    "external_yield_sync_total_ms": ("ms", "sum", True),
    "kv_blocks_end": ("blocks", "exact", False),
    "kv_blocks_start": ("blocks", "exact", False),
    "kv_blocks_used_end": ("blocks", "exact", True),
    "kv_decode_growth_blocks": ("blocks", "exact", False),
    "kv_resize_event_count": ("count", "sum", True),
    "kv_unaccounted_blocks_end": ("blocks", "exact", True),
    "length_terminated_request_count": ("count", "sum", False),
    "policy_attestation_failure_count": ("count", "sum", True),
    "prefix_cache_active_leases_end": ("leases", "exact", True),
    "prefix_cache_cached_blocks_after_prime": ("blocks", "exact", False),
    "prefix_cache_cached_blocks_end": ("blocks", "exact", False),
    "prefix_cache_enabled": ("bool", "exact", False),
    "prefix_cache_lookup_miss_count": ("count", "sum", False),
    "prefix_cache_pending_release_entries_end": ("entries", "exact", True),
    "prefix_cache_reclaim_event_count": ("count", "sum", False),
    "prefix_cache_reclaimed_block_count": ("blocks", "sum", False),
    "prefix_cache_state_bytes_end": ("bytes", "exact", True),
    "pressure_completion_token_count": ("tokens", "exact", False),
    "pressure_prompt_token_count": ("tokens", "exact", False),
    "request_count": ("count", "sum", False),
    "request_failure_count": ("count", "sum", True),
    "semantic_output_record_count": ("count", "sum", False),
    "server_exit_before_shutdown_count": ("count", "sum", True),
    "shutdown_forced_count": ("count", "sum", True),
    "shutdown_nonzero_count": ("count", "sum", True),
    "snapshot_residue_count": ("count", "sum", True),
}


class ClosureError(RuntimeError):
    pass


@dataclasses.dataclass
class Evidence:
    values: dict[str, float | int] = dataclasses.field(
        default_factory=lambda: {name: 0 for name in METRIC_DEFINITIONS}
    )
    details: dict[str, Any] = dataclasses.field(default_factory=dict)


def metric_records(values: dict[str, float | int]) -> list[dict[str, Any]]:
    if set(values) != set(METRIC_DEFINITIONS):
        missing = sorted(set(METRIC_DEFINITIONS) - set(values))
        extra = sorted(set(values) - set(METRIC_DEFINITIONS))
        raise ClosureError(f"metric set mismatch: missing={missing}, extra={extra}")
    records = []
    for name in sorted(values):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ClosureError(
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
    health: dict[str, Any],
    debug: dict[str, Any],
    binary_sha256: str,
    variant: str,
) -> list[str]:
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
        "effective_server_config_sha256": (
            provenance.get("configuration") or {}
        ).get("effective_server_config_sha256"),
        "effective_environment_sha256": (
            provenance.get("configuration") or {}
        ).get("effective_environment_sha256"),
    }
    failures = [
        f"health execution identity {field} disagrees with debug provenance"
        for field, expected in expected_summary.items()
        if identity.get(field) != expected
    ]
    for field, expected in (
        ("provenance_type", "kiln.execution-provenance.v1"),
        ("backend", BACKENDS[variant]),
        ("device", DEVICES[variant]),
        ("executable_sha256", binary_sha256),
    ):
        if identity.get(field) != expected:
            failures.append(
                f"execution identity {field}={identity.get(field)!r}, "
                f"expected {expected!r}"
            )
    for field in (
        "provenance_sha256",
        "executable_sha256",
        "numerical_runtime_sha256",
        "kernel_contract_sha256",
        "effective_server_config_sha256",
        "effective_environment_sha256",
    ):
        if re.fullmatch(r"sha256:[0-9a-f]{64}", str(identity.get(field))) is None:
            failures.append(f"execution identity {field} is not a canonical sha256")
    features = (provenance.get("kernels") or {}).get("compiled_features")
    expected_feature = BACKENDS[variant]
    if not isinstance(features, list) or expected_feature not in features:
        failures.append(
            f"execution kernel contract does not include {expected_feature}"
        )
    other_feature = "vulkan" if expected_feature == "rocm" else "rocm"
    if isinstance(features, list) and other_feature in features:
        failures.append(
            f"execution kernel contract unexpectedly includes {other_feature}"
        )
    if (provenance.get("build") or {}).get("source_dirty") is not False:
        failures.append("execution provenance does not bind a clean source tree")
    return failures


def stream_failures(
    result: mixed.StreamResult, expected_tokens: int
) -> list[str]:
    failures = []
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
    oracle_failure = mixed.deterministic_response_oracle_failure(result)
    if oracle_failure is not None:
        failures.append(f"{result.name} response oracle failed: {oracle_failure}")
    return failures


def semantic_output_sha256(results: list[mixed.StreamResult]) -> str:
    records = []
    for result in results:
        text, text_error = mixed.streamed_plain_text(result)
        records.append(
            {
                "completion_tokens": result.completion_tokens,
                "name": result.name,
                "prompt_tokens": result.prompt_tokens,
                "text": text,
                "text_error": text_error,
                "token_ids": result.token_ids,
            }
        )
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def reclaimed_block_count(events: list[mixed.ObservedEvent]) -> int:
    total = 0
    for event in events:
        if event.category != "prefix_cache_reclaim":
            continue
        value = event.fields.get("reclaimed_prefix_blocks")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ClosureError(
                "prefix-cache reclaim event omitted a positive "
                "reclaimed_prefix_blocks field"
            )
        total += value
    return total


def execute(
    model_path: Path, seed: int, variant: str, evidence: Evidence
) -> None:
    deadline = time.monotonic() + OVERALL_TIMEOUT_SECONDS
    spec = BUILD_SPECS[variant]
    binary, binary_sha256, build_seconds = mixed.build_binary(
        deadline, spec, build_timeout_seconds=BUILD_TIMEOUT_SECONDS
    )
    evidence.values["binary_build_count"] = 1
    evidence.details.update(
        {
            "backend": BACKENDS[variant],
            "build_seconds": build_seconds,
            "kiln_binary_sha256": binary_sha256,
            "variant": variant,
        }
    )
    mixed.trace(
        "backend_regression_closure_binary_built",
        backend=BACKENDS[variant],
        build_seconds=build_seconds,
        path=str(binary.relative_to(ROOT)),
        sha256=binary_sha256,
    )

    port = mixed.free_loopback_port()
    run_dir = mixed.create_serving_run_dir(variant)
    adapter_dir = run_dir / "adapters"
    snapshot_dir = run_dir / "model-snapshots"
    config_path = run_dir / "kiln.toml"
    process: Any = None
    server_log: mixed.ServerLog | None = None
    shutdown: mixed.ShutdownOutcome | None = None
    residue: list[str] = []
    failures: list[str] = []
    results: list[mixed.StreamResult] = []
    try:
        adapter_dir.mkdir(parents=True, exist_ok=False)
        mixed.write_server_config(
            config_path,
            variant,
            model_path,
            port,
            adapter_dir,
            snapshot_dir,
            rocm_graph_mode="disabled",
            kv_num_blocks=KV_NUM_BLOCKS,
            prefix_cache_max_blocks=PREFIX_CACHE_MAX_BLOCKS,
            prefix_cache_max_entries=PREFIX_CACHE_MAX_ENTRIES,
        )
        evidence.details["generated_config_sha256"] = mixed.sha256_file(config_path)
        process, server_log = mixed.start_server(
            binary, config_path, variant, spec
        )
        startup_health = mixed.wait_ready(port, process, server_log, deadline)
        startup_debug = mixed.json_request(port, "GET", "/v1/debug/model-state")
        startup_failures = [
            *mixed.attest_runtime(
                variant, startup_health, startup_debug, kv_force_blocks=None
            ),
            *execution_identity_failures(
                startup_health, startup_debug, binary_sha256, variant
            ),
        ]
        evidence.values["policy_attestation_failure_count"] = len(startup_failures)
        if startup_failures:
            raise ClosureError(
                "startup runtime attestation failed: "
                + " | ".join(startup_failures)
            )

        before_health = soak.wait_drained(
            port, deadline, "regression closure startup"
        )
        batching_before = mixed.batching_snapshot(before_health)
        prefix_before = mixed.prefix_cache_snapshot(before_health)
        if batching_before["blocks_total"] != KV_NUM_BLOCKS:
            raise ClosureError(
                f"fixed KV pool has {batching_before['blocks_total']} blocks, "
                f"expected {KV_NUM_BLOCKS}"
            )
        measurement_started = time.monotonic()

        prime = mixed.run_stream(
            port,
            name="prefix-prime",
            marker=mixed.workload_marker(seed, "prefix-prime"),
            prompt_words=PROMPT_WORDS,
            max_tokens=PRIME_MAX_TOKENS,
            seed=seed,
            absolute_deadline=deadline,
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        results.append(prime)
        failures.extend(stream_failures(prime, PRIME_MAX_TOKENS))
        prime_health = soak.wait_drained(
            port, deadline, "regression closure prefix prime"
        )
        prefix_after_prime = mixed.prefix_cache_snapshot(prime_health)

        pressure = mixed.run_stream(
            port,
            name="kv-growth-pressure",
            marker=mixed.workload_marker(seed, "kv-growth-pressure"),
            prompt_words=PROMPT_WORDS,
            max_tokens=PRESSURE_MAX_TOKENS,
            seed=seed + 1,
            absolute_deadline=deadline,
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        results.append(pressure)
        failures.extend(stream_failures(pressure, PRESSURE_MAX_TOKENS))

        final_health = soak.wait_drained(
            port, deadline, "regression closure final drain"
        )
        final_debug = mixed.json_request(port, "GET", "/v1/debug/model-state")
        batching_after = mixed.batching_snapshot(final_health)
        prefix_after = mixed.prefix_cache_snapshot(final_health)
        final_policy_failures = [
            *mixed.attest_runtime(
                variant, final_health, final_debug, kv_force_blocks=None
            ),
            *execution_identity_failures(
                final_health, final_debug, binary_sha256, variant
            ),
            *mixed.attest_runtime_execution(
                variant, before_health, final_health
            ),
        ]
        evidence.values["policy_attestation_failure_count"] += len(
            final_policy_failures
        )
        failures.extend(final_policy_failures)
        events = server_log.events_since(measurement_started)
        categories = [event.category for event in events]
        reclaim_events = [
            event for event in events if event.category == "prefix_cache_reclaim"
        ]
        reclaimed_blocks = reclaimed_block_count(reclaim_events)
        sync_values = mixed.external_yield_sync_metric_values(
            before_health, final_health
        )
        oracle_failures = sum(
            mixed.deterministic_response_oracle_failure(result) is not None
            for result in results
        )
        request_failures = sum(not result.success for result in results)
        growth_blocks = (
            (
                pressure.prompt_tokens
                + pressure.completion_tokens
                + KV_BLOCK_SIZE
                - 1
            )
            // KV_BLOCK_SIZE
            - (pressure.prompt_tokens + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
        )
        evidence.values.update(
            {
                **sync_values,
                "batching_error_count": mixed.counter_delta(
                    batching_before, batching_after, "total_errors"
                ),
                "completion_token_count": sum(
                    result.completion_tokens for result in results
                ),
                "device_fault_event_count": categories.count("device_fault"),
                "exact_output_failure_count": oracle_failures,
                "kv_blocks_end": batching_after["blocks_total"],
                "kv_blocks_start": batching_before["blocks_total"],
                "kv_blocks_used_end": batching_after["blocks_used"],
                "kv_decode_growth_blocks": growth_blocks,
                "kv_resize_event_count": categories.count("kv_resize"),
                "kv_unaccounted_blocks_end": soak.unaccounted_blocks(
                    batching_after, prefix_after
                ),
                "length_terminated_request_count": sum(
                    result.finish_reason == "length" for result in results
                ),
                "prefix_cache_active_leases_end": prefix_after["active_leases"],
                "prefix_cache_cached_blocks_after_prime": prefix_after_prime[
                    "cached_blocks"
                ],
                "prefix_cache_cached_blocks_end": prefix_after["cached_blocks"],
                "prefix_cache_enabled": int(prefix_after["enabled"]),
                "prefix_cache_lookup_miss_count": mixed.counter_delta(
                    prefix_before, prefix_after, "lookup_misses"
                ),
                "prefix_cache_pending_release_entries_end": prefix_after[
                    "pending_release_entries"
                ],
                "prefix_cache_reclaim_event_count": len(reclaim_events),
                "prefix_cache_reclaimed_block_count": reclaimed_blocks,
                "prefix_cache_state_bytes_end": prefix_after[
                    "cached_state_bytes"
                ],
                "pressure_completion_token_count": pressure.completion_tokens,
                "pressure_prompt_token_count": pressure.prompt_tokens,
                "request_count": len(results),
                "request_failure_count": request_failures,
                "semantic_output_record_count": len(results) - oracle_failures,
            }
        )
        evidence.details.update(
            {
                "prefix_cache_after_prime": prefix_after_prime,
                "prefix_cache_final": prefix_after,
                "pressure": {
                    "completion_tokens": pressure.completion_tokens,
                    "decode_growth_blocks": growth_blocks,
                    "finish_reason": pressure.finish_reason,
                    "prompt_tokens": pressure.prompt_tokens,
                },
                "reclaim_events": [
                    {
                        "reclaimed_prefix_blocks": event.fields.get(
                            "reclaimed_prefix_blocks"
                        ),
                        "requested_blocks": event.fields.get("requested_blocks"),
                    }
                    for event in reclaim_events
                ],
                "semantic_output_sha256": semantic_output_sha256(results),
            }
        )

        for name in (
            "batching_error_count",
            "device_fault_event_count",
            "exact_output_failure_count",
            "external_yield_sync_failure_count",
            "kv_resize_event_count",
            "kv_unaccounted_blocks_end",
            "policy_attestation_failure_count",
            "prefix_cache_active_leases_end",
            "prefix_cache_pending_release_entries_end",
            "request_failure_count",
        ):
            if evidence.values[name] != 0:
                failures.append(f"{name}={evidence.values[name]}, expected 0")
        for name, expected in (
            ("completion_token_count", EXPECTED_COMPLETION_TOKENS),
            ("kv_blocks_end", KV_NUM_BLOCKS),
            ("kv_blocks_start", KV_NUM_BLOCKS),
            ("length_terminated_request_count", EXPECTED_REQUESTS),
            ("pressure_completion_token_count", PRESSURE_MAX_TOKENS),
            ("request_count", EXPECTED_REQUESTS),
            ("semantic_output_record_count", EXPECTED_REQUESTS),
        ):
            if evidence.values[name] != expected:
                failures.append(
                    f"{name}={evidence.values[name]}, expected {expected}"
                )
        if growth_blocks < PRESSURE_MAX_TOKENS // KV_BLOCK_SIZE:
            failures.append(
                f"pressure request crossed only {growth_blocks} decode-growth "
                "block boundaries"
            )
        expected_prefix_enabled = variant == ROCM_VARIANT
        if prefix_after["enabled"] is not expected_prefix_enabled:
            failures.append(
                f"prefix-cache enabled={prefix_after['enabled']}, "
                f"expected {expected_prefix_enabled}"
            )
        if variant == ROCM_VARIANT:
            if prefix_after_prime["cached_blocks"] < 1:
                failures.append("ROCm prime retained no prefix-cache blocks")
            if len(reclaim_events) < 1 or reclaimed_blocks < 1:
                failures.append(
                    "ROCm pressure request did not reclaim an unleased prefix"
                )
        else:
            failures.extend(
                mixed.disabled_prefix_cache_failures(
                    prefix_after_prime, phase="Vulkan prime"
                )
            )
            failures.extend(
                mixed.disabled_prefix_cache_failures(
                    prefix_after, phase="Vulkan final"
                )
            )
            if reclaim_events or reclaimed_blocks:
                failures.append(
                    "Vulkan correctness-quarantined prefix cache reclaimed blocks"
                )
        if process.poll() is not None:
            evidence.values["server_exit_before_shutdown_count"] = 1
            failures.append(
                f"server exited before controlled shutdown ({process.returncode})"
            )
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        if process is not None:
            shutdown = mixed.terminate_process(process)
        if server_log is not None:
            server_log.join()
        residue = mixed.snapshot_payload_residue(snapshot_dir)
        if shutdown is not None:
            evidence.values["shutdown_forced_count"] = int(shutdown.forced)
            evidence.values["shutdown_nonzero_count"] = int(
                shutdown.returncode != 0
            )
            evidence.details["shutdown"] = {
                "duration_ms": shutdown.duration_ms,
                "forced": shutdown.forced,
                "returncode": shutdown.returncode,
            }
        evidence.values["snapshot_residue_count"] = len(residue)
        shutil.rmtree(run_dir, ignore_errors=True)

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
        raise ClosureError(" | ".join(dict.fromkeys(failures)))


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
    evidence = Evidence()
    status = "failed"
    details: str | None = None
    try:
        if variant not in VARIANTS:
            raise ClosureError(
                f"{VARIANT_ENV} must be one of {list(VARIANTS)!r}"
            )
        if not result_path_value:
            raise ClosureError(f"{RESULT_ENV} is required")
        model_path = args.model_path.resolve(strict=True)
        if not model_path.is_dir():
            raise ClosureError("--model-path must be a directory")
        execute(model_path, args.seed, variant, evidence)
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
        mixed.trace("backend_regression_closure_error", details=error)
    result = {
        "schema_version": 1,
        "case_id": CASE_IDS.get(variant, "backend-serving-regression-closure"),
        "status": status,
        "duration_seconds": time.monotonic() - started,
        "effective_config": EFFECTIVE_CONFIGS.get(variant, {}),
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
