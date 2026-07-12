from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import sys
import time
import unittest
from pathlib import Path
from unittest import mock


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = QUALIFICATION_DIR.parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_serve_rocm_soak",
    QUALIFICATION_DIR / "serve_rocm_soak.py",
)
assert SPEC is not None and SPEC.loader is not None
soak = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = soak
SPEC.loader.exec_module(soak)


def prefix_health(**overrides: int) -> dict:
    values = {
        "lookup_hits": 2,
        "lookup_misses": 3,
        "hit_tokens": 128,
        "hit_blocks": 2,
        "cached_blocks": 4,
        "max_blocks": 16,
        "cached_entries": 2,
        "max_entries": 8,
        "cached_state_bytes": 1024,
        "max_state_bytes": 4096,
        "active_leases": 0,
        "pending_release_entries": 0,
    }
    values.update(overrides)
    return {"prefix_cache": values}


class ServeRocmSoakTests(unittest.TestCase):
    def test_wait_drained_reads_the_nested_runtime_snapshot(self) -> None:
        health = {
            "decode_runtime": {
                "batching_engine": {
                    "active_decode": 0,
                    "active_prefill": 0,
                    "active_staged_requests": 0,
                    "queue_depth": 0,
                }
            }
        }
        with mock.patch.object(
            soak.mixed, "read_stable_health", return_value=health
        ) as read_health:
            self.assertIs(
                soak.wait_drained(8420, time.monotonic() + 1.0, "fixture"), health
            )
        read_health.assert_called_once()

    def test_prefix_cache_accounting_distinguishes_residency_from_leaks(self) -> None:
        prefix = soak.prefix_cache_snapshot(prefix_health())
        self.assertEqual(prefix["cached_blocks"], 4)
        self.assertEqual(
            soak.unaccounted_blocks({"blocks_used": 4}, prefix), 0
        )
        self.assertEqual(
            soak.unaccounted_blocks({"blocks_used": 6}, prefix), 2
        )
        with self.assertRaisesRegex(soak.SoakError, "accounts for 4 blocks"):
            soak.unaccounted_blocks({"blocks_used": 3}, prefix)
        with self.assertRaisesRegex(soak.SoakError, "exceeds max_blocks"):
            soak.prefix_cache_snapshot(prefix_health(cached_blocks=17))

    def test_wave_identity_reuses_prompts_but_changes_request_seed(self) -> None:
        with mock.patch.object(soak.mixed, "run_stream", return_value=object()) as run:
            soak.run_wave(
                8420, wave=0, base_seed=7, deadline=time.monotonic() + 1.0
            )
            first = run.call_args.kwargs
            run.reset_mock()
            soak.run_wave(
                8420, wave=4, base_seed=7, deadline=time.monotonic() + 1.0
            )
            repeated = run.call_args.kwargs
        self.assertEqual(first["marker"], repeated["marker"])
        self.assertNotEqual(first["name"], repeated["name"])
        self.assertNotEqual(first["seed"], repeated["seed"])

        with mock.patch.object(soak.mixed, "run_stream", return_value=object()) as run:
            soak.run_wave(
                8420,
                wave=0,
                base_seed=7,
                deadline=time.monotonic() + 1.0,
                phase="warmup",
                prompt_epoch=0,
            )
            warm = run.call_args.kwargs
        self.assertNotEqual(first["marker"], warm["marker"])

    def test_checked_in_workload_exactly_matches_driver_contract(self) -> None:
        path = ROOT / "qualification/workloads/serving-rocm-development-soak-v1.json"
        workload = json.loads(path.read_text())
        self.assertEqual(workload["kind"], "soak")
        self.assertIsNone(workload["comparison_policy"])
        self.assertEqual(len(workload["variants"]), 1)
        variant = workload["variants"][0]
        self.assertEqual(variant["id"], soak.RUNTIME_VARIANT)
        self.assertEqual(
            variant["effective_config"],
            soak.effective_config(
                soak.QUALIFICATION_DURATION_SECONDS,
                soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
            ),
        )
        self.assertEqual(len(variant["cases"]), 1)
        case = variant["cases"][0]
        self.assertEqual(case["id"], soak.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(soak.METRIC_DEFINITIONS),
        )
        self.assertEqual(
            case["command"],
            [
                "python3",
                "scripts/qualification/serve_rocm_soak.py",
                "--model-path",
                "${model_path}",
                "--seed",
                "${seed}",
                "--minimum-duration-seconds",
                "1800",
                "--memory-growth-limit-bytes",
                str(soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES),
            ],
        )

    def test_metric_contract_is_closed_sorted_and_finite(self) -> None:
        values = {name: 0 for name in soak.METRIC_DEFINITIONS}
        metrics = soak.metrics_from_values(values)
        self.assertEqual(
            [metric["name"] for metric in metrics], sorted(soak.METRIC_DEFINITIONS)
        )

        missing = dict(values)
        del missing["request_count"]
        with self.assertRaisesRegex(soak.SoakError, "metric set mismatch"):
            soak.metrics_from_values(missing)
        for invalid in (-1, math.inf, math.nan, True):
            with self.subTest(invalid=invalid):
                malformed = dict(values)
                malformed["request_count"] = invalid
                with self.assertRaises(soak.SoakError):
                    soak.metrics_from_values(malformed)

    def test_arguments_enforce_bounded_duration_and_growth(self) -> None:
        args = soak.parse_args(
            [
                "--model-path",
                "model",
                "--seed",
                "7",
                "--minimum-duration-seconds",
                "60",
                "--memory-growth-limit-bytes",
                "0",
            ]
        )
        self.assertEqual(args.minimum_duration_seconds, 60.0)
        self.assertEqual(args.memory_growth_limit_bytes, 0)

        for duration in ("59.999", "nan", "172801"):
            with (
                self.subTest(duration=duration),
                self.assertRaises(SystemExit),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                soak.parse_args(
                    [
                        "--model-path",
                        "model",
                        "--seed",
                        "7",
                        "--minimum-duration-seconds",
                        duration,
                    ]
                )
        with (
            self.assertRaises(SystemExit),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            soak.parse_args(
                [
                    "--model-path",
                    "model",
                    "--seed",
                    "7",
                    "--minimum-duration-seconds",
                    "60",
                    "--memory-growth-limit-bytes",
                    "-1",
                ]
            )


if __name__ == "__main__":
    unittest.main()
