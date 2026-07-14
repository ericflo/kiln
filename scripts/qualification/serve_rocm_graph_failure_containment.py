#!/usr/bin/env python3
"""Run the explicit real-ROCm graph failure-containment test corpus."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import serve_mixed_load as mixed


ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "rocm-graph-failure-containment"
VARIANT_ID = "real-rocm-graph-fault-corpus"
RESULT_ENV = mixed.RESULT_ENV
VARIANT_ENV = mixed.VARIANT_ENV
TIMEOUT_SECONDS = 1200
SERVICE_RUNTIME_MAX_SECONDS = TIMEOUT_SECONDS - 60
TESTS = (
    "shape_dependent_attention_is_cached_as_typed_eager_fallback",
    "graph_parity_across_buckets_prefix_cancellation_and_adapter_boundary",
    "stale_pool_generation_refuses_native_replay_and_falls_back_eager",
)


class FailureContainmentError(RuntimeError):
    pass


EFFECTIVE_CONFIG: dict[str, Any] = {
    "build": {
        "cargo_jobs": mixed.BUILD_CARGO_JOBS,
        "cargo_execution_mode": mixed.BUILD_CARGO_EXECUTION_MODE,
        "cargo_memory_scope": mixed.BUILD_CARGO_MEMORY_SCOPE,
        "cargo_min_available_gib": mixed.BUILD_CARGO_MIN_AVAILABLE_GIB,
        "cargo_private_network": mixed.BUILD_CARGO_PRIVATE_NETWORK,
        "cargo_service_runtime_max_seconds": SERVICE_RUNTIME_MAX_SECONDS,
        "cargo_wrapper": mixed.BUILD_CARGO_WRAPPER,
        "features": mixed.BUILD_FEATURES,
        "locked": True,
        "no_default_features": True,
        "offline": True,
        "package": "kiln-model",
        "rocm_archs": mixed.BUILD_ROCM_ARCHS,
        "rocm_path": mixed.BUILD_ROCM_PATH,
        "test_threads": 1,
        "timeout_seconds": TIMEOUT_SECONDS,
    },
    "runtime": {
        "qualification_opt_in": True,
        "required_backend": "rocm",
        "required_device": "gfx1151",
    },
    "workload": {
        "expected_test_count": len(TESTS),
        "tests": {
            f"test_{index}": name for index, name in enumerate(TESTS)
        },
    },
}


METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    "failed_test_count": ("count", "sum", True),
    "graph_lifecycle_parity_test_count": ("count", "sum", False),
    "passed_test_count": ("count", "sum", False),
    "shape_dependent_fallback_test_count": ("count", "sum", False),
    "stale_generation_containment_test_count": ("count", "sum", False),
}


def command() -> list[str]:
    return [
        str(ROOT / mixed.BUILD_CARGO_WRAPPER),
        "test",
        "--locked",
        "--offline",
        "-p",
        "kiln-model",
        "--no-default-features",
        "--features",
        "rocm",
        "--lib",
        "rocm_graph::tests::",
        "--",
        "--ignored",
        "--nocapture",
        "--test-threads=1",
    ]


def test_environment(source: dict[str, str]) -> dict[str, str]:
    environment = mixed.source_bound_build_environment(source)
    environment["KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS"] = str(
        SERVICE_RUNTIME_MAX_SECONDS
    )
    environment["KILN_QUALIFICATION"] = "1"
    return environment


def parse_passed_tests(output: str) -> set[str]:
    passed: set[str] = set()
    for name in TESTS:
        pattern = rf"test\s+rocm_graph::tests::{re.escape(name)}\s+\.{{3}}\s+ok"
        if re.search(pattern, output):
            passed.add(name)
    return passed


def metrics(passed: set[str]) -> list[dict[str, Any]]:
    values = {
        "failed_test_count": len(TESTS) - len(passed),
        "graph_lifecycle_parity_test_count": int(TESTS[1] in passed),
        "passed_test_count": len(passed),
        "shape_dependent_fallback_test_count": int(TESTS[0] in passed),
        "stale_generation_containment_test_count": int(TESTS[2] in passed),
    }
    result = []
    for name in sorted(METRIC_DEFINITIONS):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise FailureContainmentError(f"metric {name} is not finite: {value!r}")
        unit, aggregation, lower_is_better = METRIC_DEFINITIONS[name]
        result.append(
            {
                "name": name,
                "value": value,
                "unit": unit,
                "aggregation": aggregation,
                "lower_is_better": lower_is_better,
            }
        )
    return result


def execute() -> tuple[list[dict[str, Any]], str | None]:
    completed = subprocess.run(
        command(),
        cwd=ROOT,
        env=test_environment(dict(os.environ)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    output = completed.stdout
    passed = parse_passed_tests(output)
    failures: list[str] = []
    if completed.returncode != 0:
        failures.append(f"bounded ROCm graph test process returned {completed.returncode}")
    missing = [name for name in TESTS if name not in passed]
    if missing:
        failures.append("missing passing tests: " + ", ".join(missing))
    if not re.search(r"test result:\s+ok\.\s+3 passed;\s+0 failed", output):
        failures.append("Cargo summary did not report exactly 3 passed and 0 failed")
    if "[rocm-graph-parity]" not in output:
        failures.append("graph lifecycle parity evidence marker is missing")
    if "[rocm-stale-generation]" not in output:
        failures.append("stale-generation containment evidence marker is missing")
    if re.search(r"(?i)(no rocm device|skipp(?:ed|ing).*rocm)", output):
        failures.append("required ROCm execution was skipped or unavailable")
    if failures:
        tail = output[-4000:].replace("\n", " | ")
        failures.append(f"bounded output tail: {tail}")
    return metrics(passed), " | ".join(failures) if failures else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    parse_args(argv)
    variant = os.environ.get(VARIANT_ENV, "")
    result_path_value = os.environ.get(RESULT_ENV)
    if not result_path_value:
        print(f"{RESULT_ENV} is required", file=os.sys.stderr)
        return 2
    status = "failed"
    details: str | None = None
    values = metrics(set())
    try:
        if variant != VARIANT_ID:
            raise FailureContainmentError(
                f"{VARIANT_ENV} must be {VARIANT_ID!r}, got {variant!r}"
            )
        values, details = execute()
        status = "passed" if details is None else "failed"
    except Exception as exc:
        details = f"{type(exc).__name__}: {exc}"
        mixed.trace("graph_failure_containment_error", details=details)
    result = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "status": status,
        "duration_seconds": time.monotonic() - started,
        "effective_config": EFFECTIVE_CONFIG,
        "metrics": values,
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
