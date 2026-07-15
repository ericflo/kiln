from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = QUALIFICATION_DIR.parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_serve_development_soak",
    QUALIFICATION_DIR / "serve_development_soak.py",
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
    def test_runtime_profiles_are_closed_and_backend_specific(self) -> None:
        self.assertIs(
            soak.runtime_for_variant(soak.ROCM_RUNTIME.variant_id), soak.ROCM_RUNTIME
        )
        self.assertIs(
            soak.runtime_for_variant(soak.VULKAN_RUNTIME.variant_id),
            soak.VULKAN_RUNTIME,
        )
        with self.assertRaisesRegex(soak.SoakError, "must name one of"):
            soak.runtime_for_variant("unknown")

        vulkan = soak.effective_config(
            soak.QUALIFICATION_DURATION_SECONDS,
            soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
            soak.VULKAN_RUNTIME,
        )
        self.assertEqual(vulkan["build"]["features"], "vulkan")
        self.assertFalse(vulkan["runtime"]["rocm_graphs_enabled"])
        self.assertEqual(vulkan["server"]["request_timeout_seconds"], 600)
        self.assertEqual(vulkan["soak"]["gpu_memory_scope"], "server_process")
        self.assertEqual(vulkan["soak"]["stabilization_min_cycles"], 4)
        self.assertEqual(vulkan["soak"]["stabilization_max_cycles"], 8)
        self.assertEqual(
            vulkan["soak"]["host_mem_available_floor_bytes"], 8 * 1024**3
        )
        self.assertNotIn(
            "host_mem_available_floor_bytes",
            soak.effective_config(
                soak.QUALIFICATION_DURATION_SECONDS,
                soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
            )["soak"],
        )

    def test_process_memory_snapshot_requires_and_converts_linux_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            status = root / "42" / "status"
            status.parent.mkdir(parents=True)
            status.write_text(
                "Name:\tkiln\n"
                "VmRSS:\t1000 kB\n"
                "RssAnon:\t700 kB\n"
                "RssFile:\t250 kB\n"
                "RssShmem:\t50 kB\n"
                "VmSwap:\t25 kB\n",
                encoding="utf-8",
            )
            snapshot = soak.process_memory_snapshot(42, root)
        self.assertEqual(snapshot.rss_bytes, 1000 * 1024)
        self.assertEqual(snapshot.rss_anon_bytes, 700 * 1024)
        self.assertEqual(snapshot.rss_file_bytes, 250 * 1024)
        self.assertEqual(snapshot.rss_shmem_bytes, 50 * 1024)
        self.assertEqual(snapshot.swap_bytes, 25 * 1024)

    def test_process_drm_memory_deduplicates_client_ids_and_regions(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fdinfo = root / "42" / "fdinfo"
            fdinfo.mkdir(parents=True)
            (fdinfo / "9").write_text(
                "drm-client-id:\t7\n"
                "drm-memory-vram:\t100 KiB\n"
                "drm-memory-gtt:\t20 KiB\n"
                "drm-memory-cpu:\t0 KiB\n",
                encoding="utf-8",
            )
            (fdinfo / "11").write_text(
                "drm-client-id:\t7\n"
                "drm-memory-vram:\t101 KiB\n"
                "drm-memory-gtt:\t19 KiB\n"
                "drm-memory-cpu:\t0 KiB\n",
                encoding="utf-8",
            )
            (fdinfo / "12").write_text(
                "drm-client-id:\t8\n"
                "drm-memory-vram:\t3 KiB\n"
                "drm-memory-gtt:\t4 KiB\n"
                "drm-memory-cpu:\t5 KiB\n",
                encoding="utf-8",
            )
            (fdinfo / "13").write_text("pos:\t0\n", encoding="utf-8")
            observed = soak.process_drm_memory_bytes(42, root)
        self.assertEqual(observed, (101 + 20 + 3 + 4 + 5) * 1024)

    def test_process_drm_memory_fails_closed_without_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fdinfo = root / "42" / "fdinfo"
            fdinfo.mkdir(parents=True)
            (fdinfo / "1").write_text("pos:\t0\n", encoding="utf-8")
            with self.assertRaisesRegex(soak.SoakError, "no DRM memory accounting"):
                soak.process_drm_memory_bytes(42, root)

    def test_gpu_memory_scope_selects_process_or_device_accounting(self) -> None:
        with (
            mock.patch.object(soak, "process_drm_memory_bytes", return_value=123) as drm,
            mock.patch.object(soak.mixed, "text_request", return_value="") as metrics,
        ):
            self.assertEqual(
                soak.gpu_memory_bytes(8420, 42, soak.VULKAN_RUNTIME), 123
            )
        drm.assert_called_once_with(42)
        metrics.assert_not_called()

        with (
            mock.patch.object(
                soak.mixed,
                "text_request",
                return_value='kiln_gpu_memory_bytes{kind="used"} 456\n',
            ),
            mock.patch.object(
                soak.mixed, "parse_prometheus_used_bytes", return_value=456
            ),
        ):
            self.assertEqual(soak.gpu_memory_bytes(8420, 42, soak.ROCM_RUNTIME), 456)

    def test_gpu_memory_sampler_uses_the_declared_runtime_scope(self) -> None:
        sampler = soak.GpuMemorySampler(8420, 42, soak.VULKAN_RUNTIME)
        with mock.patch.object(soak, "gpu_memory_bytes", return_value=789) as sample:
            sampler._sample()
        self.assertEqual(sampler.samples, [789])
        self.assertEqual(sampler.errors, [])
        sample.assert_called_once_with(8420, 42, soak.VULKAN_RUNTIME)

    def test_graph_warmup_contract_depends_on_runtime(self) -> None:
        graph = {"capture_successes": 1, "replay_successes": 1, "failures": 0}
        self.assertTrue(soak.graph_warmup_ready(graph, soak.ROCM_RUNTIME))
        self.assertFalse(soak.graph_warmup_ready(graph, soak.VULKAN_RUNTIME))
        self.assertTrue(
            soak.graph_warmup_ready({name: 0 for name in graph}, soak.VULKAN_RUNTIME)
        )

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

        with mock.patch.object(soak.mixed, "run_stream", return_value=object()) as run:
            soak.run_wave(
                8420,
                wave=2,
                base_seed=7,
                deadline=time.monotonic() + 1.0,
                runtime=soak.VULKAN_RUNTIME,
            )
        self.assertEqual(run.call_count, 8)
        self.assertTrue(
            all(call.kwargs["max_tokens"] == 16 for call in run.call_args_list)
        )
        self.assertTrue(
            all(
                call.kwargs["request_timeout_seconds"] == 600.0
                for call in run.call_args_list
            )
        )

    def test_cancellation_markers_are_unique_and_confirmation_is_required(self) -> None:
        def result(*_args: object, **kwargs: object) -> mock.Mock:
            return mock.Mock(
                cancelled=True,
                semantic_times=[0.0] * soak.mixed.CANCELLATION_AFTER_DELTAS,
                marker=kwargs["marker"],
            )

        with (
            mock.patch.object(soak.mixed, "run_stream", side_effect=result) as run,
            mock.patch.object(
                soak.mixed,
                "wait_for_cancellation_and_drain",
                return_value=(True, {}),
            ),
        ):
            self.assertIsNone(
                soak.run_cancellation(
                    8420,
                    wave=4,
                    base_seed=7,
                    phase="measured",
                    deadline=time.monotonic() + 1.0,
                )
            )
            first_marker = run.call_args.kwargs["marker"]
            self.assertIsNone(
                soak.run_cancellation(
                    8420,
                    wave=9,
                    base_seed=7,
                    phase="measured",
                    deadline=time.monotonic() + 1.0,
                )
            )
            repeated_marker = run.call_args.kwargs["marker"]
        self.assertNotEqual(first_marker, repeated_marker)

        with (
            mock.patch.object(soak.mixed, "run_stream", side_effect=result),
            mock.patch.object(
                soak.mixed,
                "wait_for_cancellation_and_drain",
                return_value=(False, {}),
            ),
        ):
            self.assertIn(
                "not confirmed",
                soak.run_cancellation(
                    8420,
                    wave=4,
                    base_seed=7,
                    phase="measured",
                    deadline=time.monotonic() + 1.0,
                ),
            )

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

    def test_checked_in_vulkan_workload_exactly_matches_driver_contract(self) -> None:
        path = ROOT / "qualification/workloads/serving-vulkan-development-soak-v1.json"
        workload = json.loads(path.read_text())
        self.assertEqual(workload["kind"], "soak")
        self.assertIsNone(workload["comparison_policy"])
        self.assertEqual(len(workload["variants"]), 1)
        variant = workload["variants"][0]
        self.assertEqual(variant["id"], soak.VULKAN_RUNTIME.variant_id)
        self.assertEqual(variant["backend"], "vulkan")
        self.assertEqual(
            variant["effective_config"],
            soak.effective_config(
                soak.QUALIFICATION_DURATION_SECONDS,
                soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
                soak.VULKAN_RUNTIME,
            ),
        )
        case = variant["cases"][0]
        self.assertEqual(case["id"], soak.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(soak.metric_definitions(soak.VULKAN_RUNTIME)),
        )
        self.assertEqual(
            case["command"],
            [
                "python3",
                "scripts/qualification/serve_development_soak.py",
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
        self.assertGreaterEqual(
            case["timeout_seconds"],
            soak.QUALIFICATION_DURATION_SECONDS
            + soak.VULKAN_RUNTIME.setup_deadline_seconds,
        )

    def test_host_memory_guard_terminates_below_the_floor_and_records_pressure(
        self,
    ) -> None:
        process = mock.Mock(pid=1234)
        process.poll.return_value = None
        guard = soak.HostMemoryGuard(process, 8 * 1024**3)
        with (
            mock.patch.object(
                soak,
                "host_memory_snapshot",
                side_effect=[(10 * 1024**3, 100), (7 * 1024**3, 250)],
            ),
            mock.patch.object(soak.os, "killpg") as killpg,
        ):
            guard._sample()
            guard._sample()
        killpg.assert_called_once_with(1234, signal.SIGTERM)
        self.assertIn("below", guard.trip_reason or "")
        self.assertEqual(guard.metric_values()["host_memory_guard_trip_count"], 1)
        self.assertEqual(guard.metric_values()["host_swap_growth_bytes"], 150)
        self.assertEqual(
            guard.metric_values()["host_mem_available_min_bytes"], 7 * 1024**3
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

        vulkan_values = {
            name: 0 for name in soak.metric_definitions(soak.VULKAN_RUNTIME)
        }
        vulkan_metrics = soak.metrics_from_values(
            vulkan_values, soak.VULKAN_RUNTIME
        )
        self.assertEqual(
            [metric["name"] for metric in vulkan_metrics], sorted(vulkan_values)
        )

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
