from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = QUALIFICATION_DIR.parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_serve_rocm_pressure",
    QUALIFICATION_DIR / "serve_rocm_pressure.py",
)
assert SPEC is not None and SPEC.loader is not None
pressure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pressure
SPEC.loader.exec_module(pressure)


def prometheus_text(**overrides: float) -> str:
    values = {
        'kiln_gpu_memory_bytes{kind="free"}': 8,
        'kiln_gpu_memory_bytes{kind="total"}': 100,
        'kiln_memory_reclaim_attempts_total{outcome="reclaimed"}': 0,
        'kiln_memory_reclaim_attempts_total{outcome="zero_yield"}': 2,
        "kiln_memory_reclaim_suppressed_total": 4,
        "kiln_memory_reclaimed_bytes_total": 0,
        'kiln_memory_reclaim_last_bytes{kind="target"}': 17,
        'kiln_memory_reclaim_last_bytes{kind="reclaimed"}': 0,
        "kiln_memory_reclaim_last_duration_seconds": 0.000125,
        "kiln_memory_reclaim_retry_after_seconds": 4,
        "kiln_memory_reclaim_zero_yield_streak": 2,
    }
    values.update(overrides)
    return "\n".join(f"{selector} {value}" for selector, value in values.items()) + "\n"


def governor_health(**overrides: int) -> dict:
    values = {
        "automatic_attempts": 2,
        "automatic_successful_attempts": 0,
        "automatic_zero_yield_attempts": 2,
        "automatic_suppressed_attempts": 4,
        "automatic_reclaimed_bytes": 0,
        "automatic_last_target_bytes": 17,
        "automatic_last_reclaimed_bytes": 0,
        "automatic_last_duration_us": 125,
        "automatic_retry_after_ms": 4000,
        "automatic_zero_yield_streak": 2,
    }
    values.update(overrides)
    return {"decode_runtime": {"memory_governor": values}}


def stream_result(ready_times: list[float]) -> pressure.mixed.StreamResult:
    return pressure.mixed.StreamResult(
        name="fixture",
        marker="fixture",
        started=0.0,
        finished=10.0,
        semantic_times=[1.0],
        token_ready_times=ready_times,
        token_queue_delays_ms=[0.0] * len(ready_times),
        prompt_tokens=1,
        completion_tokens=len(ready_times),
        usage_records=1,
        finish_reason="length",
        done=True,
        cancelled=False,
        error=None,
    )


def ready_payload() -> dict:
    return {
        "schema_version": 1,
        "pid": 123,
        "allocated_bytes": 92,
        "target_free_fraction": pressure.TARGET_FREE_FRACTION,
        "minimum_free_fraction": pressure.MINIMUM_FREE_FRACTION,
        "baseline": {
            "total_bytes": 100,
            "used_bytes": 10,
            "free_bytes": 90,
            "free_fraction": 0.9,
        },
        "ready": {
            "total_bytes": 100,
            "used_bytes": 92,
            "free_bytes": 8,
            "free_fraction": 0.08,
        },
    }


class ServeRocmPressureTests(unittest.TestCase):
    def test_checked_in_workload_exactly_matches_driver_contract(self) -> None:
        path = (
            ROOT
            / "qualification/workloads/serving-rocm-memory-pressure-v1.json"
        )
        workload = json.loads(path.read_text())
        self.assertEqual(workload["determinism"]["seed_delivery"], "argv")
        self.assertEqual(len(workload["variants"]), 1)
        variant = workload["variants"][0]
        self.assertEqual(variant["id"], pressure.VARIANT_ID)
        self.assertEqual(variant["effective_config"], pressure.EFFECTIVE_CONFIG)
        self.assertEqual(len(variant["cases"]), 1)
        case = variant["cases"][0]
        self.assertEqual(case["id"], pressure.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(pressure.METRIC_DEFINITIONS),
        )
        self.assertEqual(
            case["command"],
            [
                "python3",
                "scripts/qualification/serve_rocm_pressure.py",
                "--model-path",
                "${model_path}",
                "--seed",
                "${seed}",
            ],
        )

    def test_prometheus_parser_is_closed_and_computes_free_fraction(self) -> None:
        values = pressure.parse_prometheus_values(prometheus_text())
        self.assertEqual(pressure.memory_free_fraction(values), 0.08)

        with self.assertRaisesRegex(pressure.mixed.QualificationError, "missing"):
            pressure.parse_prometheus_values(
                prometheus_text().replace("kiln_memory_reclaim_zero_yield_streak 2\n", "")
            )
        with self.assertRaisesRegex(pressure.mixed.QualificationError, "more than once"):
            pressure.parse_prometheus_values(
                prometheus_text() + "kiln_memory_reclaim_zero_yield_streak 2\n"
            )
        with self.assertRaisesRegex(pressure.mixed.QualificationError, "nonnegative"):
            pressure.parse_prometheus_values(
                prometheus_text(
                    **{"kiln_memory_reclaim_suppressed_total": -1}
                )
            )

    def test_governor_snapshot_enforces_outcome_invariants(self) -> None:
        snapshot = pressure.governor_snapshot(governor_health())
        self.assertEqual(snapshot["automatic_attempts"], 2)
        with self.assertRaisesRegex(pressure.mixed.QualificationError, "do not equal"):
            pressure.governor_snapshot(governor_health(automatic_attempts=3))
        with self.assertRaisesRegex(pressure.mixed.QualificationError, "cumulative"):
            pressure.governor_snapshot(
                governor_health(
                    automatic_successful_attempts=1,
                    automatic_zero_yield_attempts=1,
                    automatic_reclaimed_bytes=1,
                    automatic_last_reclaimed_bytes=2,
                )
            )

    def test_prometheus_and_health_must_agree(self) -> None:
        metrics = pressure.parse_prometheus_values(prometheus_text())
        health = pressure.governor_snapshot(governor_health())
        self.assertEqual(pressure.prometheus_health_mismatches(metrics, health), [])
        health["automatic_suppressed_attempts"] += 1
        self.assertTrue(pressure.prometheus_health_mismatches(metrics, health))

    def test_pressure_itl_gaps_require_both_endpoints_inside_window(self) -> None:
        result = stream_result([0.5, 1.0, 1.4, 2.0, 2.2, 3.0])
        observed = pressure.pressure_itl_gaps([result], 1.0, 2.5)
        self.assertEqual(len(observed), 3)
        for actual, expected in zip(observed, (400.0, 600.0, 200.0)):
            self.assertAlmostEqual(actual, expected)

    def test_ready_payload_is_strict_and_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ready.json"
            path.write_text(json.dumps(ready_payload()))
            self.assertEqual(pressure.load_ready_file(path)["allocated_bytes"], 92)

            malformed = ready_payload()
            malformed["ready"]["free_fraction"] = 0.07
            path.write_text(json.dumps(malformed))
            with self.assertRaisesRegex(pressure.mixed.QualificationError, "inconsistent"):
                pressure.load_ready_file(path)

            malformed = ready_payload()
            malformed["ready"]["free_bytes"] = 4
            malformed["ready"]["used_bytes"] = 96
            malformed["ready"]["free_fraction"] = 0.04
            path.write_text(json.dumps(malformed))
            with self.assertRaisesRegex(pressure.mixed.QualificationError, "readiness at"):
                pressure.load_ready_file(path)

    def test_metric_contract_is_closed_sorted_and_finite(self) -> None:
        metrics = pressure.zero_metrics()
        self.assertEqual(
            [metric["name"] for metric in metrics], sorted(pressure.METRIC_DEFINITIONS)
        )
        values = {name: 0 for name in pressure.METRIC_DEFINITIONS}
        del values["request_count"]
        with self.assertRaisesRegex(pressure.mixed.QualificationError, "mismatch"):
            pressure.metrics_from_values(values)


if __name__ == "__main__":
    unittest.main()
