from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from scripts.qualification.tests.generated_toml import parse_generated_toml


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


def batched_state_debug(
    *,
    resident_overrides: dict[str, int] | None = None,
    **overrides: int | bool,
) -> dict:
    values: dict[str, int | bool] = {
        "entry_present": True,
        "capacity_rows": 8,
        "logical_rows": 8,
        "resident": True,
        "active_leases": 0,
        "max_active_leases": 1,
        **{field: 0 for field in soak.BATCHED_STATE_CACHE_COUNTER_FIELDS},
    }
    values.update(overrides)
    resident = {"entry_count": 0, "buffer_bytes": 0, "allocation_bytes": 0}
    resident.update(resident_overrides or {})
    return {
        "caches": {
            "batched_recurrent_state": values,
            "resident_recurrent_state": resident,
        }
    }


def write_amd_telemetry_fixture(root: Path) -> dict[str, Path]:
    device = root / "card1/device"
    hwmon = device / "hwmon/hwmon6"
    hwmon.mkdir(parents=True)
    paths = {
        "vendor": device / "vendor",
        "busy": device / "gpu_busy_percent",
        "dpm": device / "pp_dpm_sclk",
        "name": hwmon / "name",
        "sclk_label": hwmon / "freq1_label",
        "sclk": hwmon / "freq1_input",
        "power_label": hwmon / "power1_label",
        "power": hwmon / "power1_average",
    }
    values = {
        "vendor": "0x1002\n",
        "busy": "80\n",
        "dpm": "0: 600Mhz\n1: 947Mhz *\n2: 2900Mhz\n",
        "name": "amdgpu\n",
        "sclk_label": "sclk\n",
        "sclk": "2400000000\n",
        "power_label": "PPT\n",
        "power": "45000000\n",
    }
    for name, value in values.items():
        paths[name].write_text(value, encoding="utf-8")
    return paths


class ServeRocmSoakTests(unittest.TestCase):
    def test_drained_warmup_evidence_survives_later_policy_failure(self) -> None:
        requests, waves = soak.record_drained_warmup_wave(
            9,
            2,
            [mock.sentinel.result] * 12,
        )
        self.assertEqual((requests, waves), (21, 3))

    def test_invalid_stream_diagnostic_names_every_contract_dimension(self) -> None:
        result = soak.mixed.StreamResult(
            name="soak-stabilize-w00010-r01",
            marker="QUAL-7-soak-shared-r01",
            started=1.0,
            finished=2.0,
            semantic_times=[],
            token_ready_times=[1.5],
            token_queue_delays_ms=[0.0],
            prompt_tokens=191,
            completion_tokens=1,
            usage_records=0,
            finish_reason="stop",
            done=False,
            cancelled=True,
            error="QualificationError: timing mismatch",
            token_ids=[151643, 151644],
            resident_prefill_used=True,
            actor_queue_ms=1.25,
            actor_admission_ms=2.5,
            actor_prefill_wall_ms=3.75,
            semantic_deltas=[
                {"choices": [{"delta": {"content": "000>0:0:5"}}]}
            ],
        )

        summary = soak.invalid_stream_results_summary([result], 16)

        for expected in (
            "soak-stabilize-w00010-r01",
            "success=false+finish_reason!=length+completion_tokens!=16+response_oracle_failed",
            "QualificationError: timing mismatch",
            "finish_reason='stop'",
            "prompt_tokens=191",
            "completion_tokens=1",
            "usage_records=0",
            "token_timings=1",
            "token_ids=[151643, 151644]",
            "resident_prefill_used=True",
            "semantic_events=0",
            "done=False",
            "cancelled=True",
            "response_oracle_failure='plain-text output is not a prefix",
            "response_text='000>0:0:5'",
            "actor_queue_ms=1.25",
            "actor_admission_ms=2.5",
            "actor_prefill_wall_ms=3.75",
        ):
            self.assertIn(expected, summary)

    def test_stream_result_contract_requires_the_deterministic_oracle(self) -> None:
        def result(content: str) -> soak.mixed.StreamResult:
            return soak.mixed.StreamResult(
                name="soak-measured-w00005-r02",
                marker="QUAL-7-soak-shared-r02",
                started=1.0,
                finished=2.0,
                semantic_times=[1.1],
                token_ready_times=[1.1] * 16,
                token_queue_delays_ms=[0.0] * 16,
                prompt_tokens=191,
                completion_tokens=16,
                usage_records=1,
                finish_reason="length",
                done=True,
                cancelled=False,
                error=None,
                semantic_deltas=[
                    {"choices": [{"delta": {"content": content}}]}
                ],
            )

        self.assertTrue(soak.valid_stream_result(result("000000 000001 00"), 16))
        corrupt = result("000000 000000 00")
        self.assertFalse(soak.valid_stream_result(corrupt, 16))
        self.assertEqual(
            soak.stream_result_violations(corrupt, 16),
            ["response_oracle_failed"],
        )

    def test_measurement_result_evidence_retains_only_completed_results(self) -> None:
        def result(name: str, *, error: str | None = None) -> soak.mixed.StreamResult:
            return soak.mixed.StreamResult(
                name=name,
                marker=f"QUAL-7-{name}",
                started=1.0,
                finished=2.0,
                semantic_times=[1.1],
                token_ready_times=[1.1 + index * 0.01 for index in range(16)],
                token_queue_delays_ms=[0.0] * 16,
                prompt_tokens=191,
                completion_tokens=16,
                usage_records=1,
                finish_reason="length",
                done=True,
                cancelled=False,
                error=error,
                latency_phases={
                    **{
                        f"{phase}_ms": None
                        for phase in soak.mixed.LATENCY_PHASE_NAMES
                    },
                    "decode_ms": 125.0,
                    "sampling_ms": 0.0,
                },
                semantic_deltas=[
                    {"choices": [{"delta": {"content": "000000 000001 00"}}]}
                ],
            )

        evidence = soak.measurement_result_evidence(
            warmup_itl_ms=[10.0] * 15,
            results=[result("complete"), result("failed", error="non-finite output")],
            measurement_events=[],
            all_server_events=[
                soak.mixed.ObservedEvent(2.0, "device_fault", "fixture fault")
            ],
            expected_completion_tokens=16,
            cancellation_count=3,
            duration_seconds=458.0,
            wave_count=6,
            steady_state_warmup_request_count=30,
            steady_state_warmup_wave_count=6,
        )

        self.assertEqual([item.name for item in evidence.successes], ["complete"])
        self.assertEqual(evidence.attributed_itl_outliers, 0)
        self.assertEqual(evidence.unexplained_itl_outliers, 0)
        self.assertEqual(evidence.values["latency_phase_metadata_missing_count"], 0)
        self.assertEqual(evidence.values["latency_phase_decode_ms_total"], 125.0)
        self.assertEqual(evidence.values["latency_phase_decode_request_count"], 1)
        self.assertEqual(evidence.values["latency_phase_sampling_ms_total"], 0.0)
        self.assertEqual(evidence.values["latency_phase_sampling_request_count"], 1)
        self.assertEqual(
            {
                name: evidence.values[name]
                for name in (
                    "cancellation_confirmed_count",
                    "completion_token_count",
                    "device_fault_event_count",
                    "non_finite_response_count",
                    "output_token_throughput_per_second",
                    "request_count",
                    "request_failure_count",
                    "soak_duration_seconds",
                    "steady_state_warmup_request_count",
                    "steady_state_warmup_wave_count",
                    "wave_count",
                )
            },
            {
                "cancellation_confirmed_count": 3,
                "completion_token_count": 16,
                "device_fault_event_count": 1,
                "non_finite_response_count": 1,
                "output_token_throughput_per_second": 16 / 458.0,
                "request_count": 2,
                "request_failure_count": 1,
                "soak_duration_seconds": 458.0,
                "steady_state_warmup_request_count": 30,
                "steady_state_warmup_wave_count": 6,
                "wave_count": 6,
            },
        )

    def test_runtime_profiles_are_closed_and_backend_specific(self) -> None:
        self.assertIs(
            soak.runtime_for_variant(soak.ROCM_RUNTIME.variant_id), soak.ROCM_RUNTIME
        )
        self.assertIs(
            soak.runtime_for_variant(soak.VULKAN_RUNTIME.variant_id),
            soak.VULKAN_RUNTIME,
        )
        self.assertIs(
            soak.runtime_for_variant(soak.VULKAN_ENDURANCE_RUNTIME.variant_id),
            soak.VULKAN_ENDURANCE_RUNTIME,
        )
        self.assertIs(
            soak.runtime_for_variant(soak.CUDA_ENDURANCE_RUNTIME.variant_id),
            soak.CUDA_ENDURANCE_RUNTIME,
        )
        self.assertIs(
            soak.runtime_for_variant(soak.METAL_ENDURANCE_RUNTIME.variant_id),
            soak.METAL_ENDURANCE_RUNTIME,
        )
        with self.assertRaisesRegex(soak.SoakError, "must name one of"):
            soak.runtime_for_variant("unknown")

        rocm = soak.effective_config(
            soak.QUALIFICATION_DURATION_SECONDS,
            soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
            soak.ROCM_RUNTIME,
        )
        self.assertEqual(
            rocm["server"]["max_active_requests"], soak.mixed.MAX_ACTIVE_REQUESTS
        )
        self.assertEqual(rocm["server"]["max_batch_tokens"], 512)
        self.assertEqual(rocm["server"]["max_prefill_tokens_per_cycle"], 256)
        self.assertEqual(
            rocm["server"]["max_prefill_layers_per_cycle"],
            soak.mixed.MAX_PREFILL_LAYERS_PER_CYCLE,
        )
        self.assertEqual(
            rocm["batching"]["actor_cycle_idle_ms"],
            soak.mixed.ACTOR_CYCLE_IDLE_MS,
        )
        self.assertEqual(
            rocm["soak"]["rocm_graph_cache_entries"], soak.ROCM_GRAPH_CACHE_MAX
        )
        self.assertEqual(
            rocm["soak"]["rocm_graph_admission_policy"],
            "idle_owner_lru_then_active_fair_lru",
        )
        self.assertEqual(rocm["soak"]["rocm_graph_active_owner_floor"], 1)
        self.assertEqual(
            rocm["soak"]["rocm_graph_transition_headroom_entries"], 0
        )
        self.assertEqual(
            rocm["soak"]["host_mem_available_floor_bytes"], 8 * 1024**3
        )
        self.assertEqual(
            rocm["soak"]["host_swap_growth_limit_bytes"], 512 * 1024**2
        )
        self.assertEqual(
            rocm["soak"]["accelerator_telemetry"],
            {
                "active_busy_floor_percent": 50,
                "amd_gpu_vendor_id": "0x1002",
                "device_selector": "exactly_one_amd_drm_device",
                "mode": "required",
                "poll_interval_ms": 250,
                "sources": {
                    "busy": "drm_device/gpu_busy_percent",
                    "power": "amdgpu_hwmon/power_PPT_average",
                    "sclk": "amdgpu_hwmon/freq_sclk_input",
                    "sclk_advertised_max": "drm_device/pp_dpm_sclk",
                },
            },
        )
        with self.assertRaisesRegex(
            soak.SoakError, "one protected geometry.*transition headroom"
        ):
            soak.effective_config(
                soak.QUALIFICATION_DURATION_SECONDS,
                soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
                soak.dataclasses.replace(soak.ROCM_RUNTIME, graph_cache_max=11),
            )

        vulkan = soak.effective_config(
            soak.QUALIFICATION_DURATION_SECONDS,
            soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
            soak.VULKAN_RUNTIME,
        )
        self.assertEqual(vulkan["build"]["features"], "vulkan")
        self.assertFalse(vulkan["runtime"]["rocm_graphs_enabled"])
        self.assertEqual(vulkan["server"]["request_timeout_seconds"], 600)
        self.assertEqual(vulkan["server"]["max_active_requests"], 4)
        self.assertEqual(vulkan["server"]["max_prefill_staging_slots"], 2)
        self.assertEqual(vulkan["server"]["max_decode_batch"], 2)
        self.assertEqual(
            vulkan["server"]["max_prefill_layers_per_cycle"],
            soak.VULKAN_MAX_PREFILL_LAYERS_PER_CYCLE,
        )
        self.assertEqual(vulkan["batching"]["actor_cycle_idle_ms"], 0)
        self.assertEqual(vulkan["batching"]["prefill_admission_quantum"], 2)
        self.assertEqual(vulkan["runtime"]["vulkan_buffer_pool_gb"], 3.5)
        self.assertEqual(vulkan["soak"]["vulkan_buffer_pool_gb"], 3.5)
        self.assertEqual(
            vulkan["soak"]["wave_concurrency"], {"wave_0": 1, "wave_1": 4}
        )
        self.assertEqual(
            vulkan["soak"]["prompt_words"],
            {"slot_0": 16, "slot_1": 32, "slot_2": 64, "slot_3": 96},
        )
        self.assertEqual(vulkan["soak"]["prompt_assignment"], "cohort_by_cycle")
        self.assertEqual(
            vulkan["soak"]["prompt_identity"],
            "fixed_by_cycle_cohort_measured_unique_by_epoch_warmup",
        )
        self.assertEqual(
            vulkan["soak"]["deadline_policy"],
            "independent_build_setup_and_measurement",
        )
        self.assertEqual(vulkan["soak"]["measurement_deadline_seconds"], 2400)
        self.assertEqual(vulkan["soak"]["qualification_case_timeout_seconds"], 5280)
        self.assertEqual(vulkan["soak"]["teardown_grace_seconds"], 180)
        self.assertEqual(vulkan["soak"]["gpu_memory_scope"], "server_process")
        self.assertEqual(vulkan["soak"]["stabilization_min_cycles"], 4)
        self.assertEqual(vulkan["soak"]["stabilization_max_cycles"], 8)
        self.assertEqual(
            vulkan["soak"]["active_gpu_peak_growth_limit_bytes"], 1024**3
        )
        self.assertEqual(vulkan["soak"]["vulkan_allocation_growth_limit_count"], 0)
        self.assertEqual(
            vulkan["soak"]["host_mem_available_floor_bytes"], 8 * 1024**3
        )
        self.assertEqual(
            vulkan["soak"]["accelerator_telemetry"]["mode"], "if_available"
        )
        cuda = soak.effective_config(
            soak.CUDA_ENDURANCE_DURATION_SECONDS,
            soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
            soak.CUDA_ENDURANCE_RUNTIME,
        )
        self.assertEqual(cuda["build"]["features"], "cuda")
        self.assertTrue(cuda["build"]["qualification_device_required"])
        self.assertNotIn("cuda_archs", cuda["build"])
        self.assertNotIn("cudarc_cuda_version", cuda["build"])
        self.assertEqual(cuda["runtime"]["serving_profile"], "stable")
        self.assertFalse(cuda["runtime"]["prefix_cache_requested_enabled"])
        self.assertFalse(cuda["runtime"]["prefix_cache_effective_enabled"])
        self.assertEqual(
            cuda["runtime"]["prefix_cache_effective_reason"],
            "cuda_prefill_semantics_quarantine",
        )
        self.assertEqual(cuda["server"]["max_decode_batch"], 4)
        self.assertEqual(cuda["server"]["max_prefill_staging_slots"], 0)
        self.assertEqual(cuda["server"]["max_active_requests"], 4)
        self.assertEqual(cuda["server"]["max_prefill_staging_priority_burst"], 0)
        self.assertEqual(
            cuda["soak"]["wave_concurrency"], {"wave_0": 1, "wave_1": 4}
        )
        self.assertEqual(cuda["soak"]["stabilization_min_cycles"], 8)
        self.assertEqual(cuda["soak"]["stabilization_max_cycles"], 16)
        self.assertEqual(cuda["soak"]["stabilization_required_stable_cycles"], 4)
        self.assertEqual(
            cuda["soak"]["gpu_memory_baseline_mode"],
            "stabilization_envelope_high_water",
        )
        self.assertEqual(cuda["soak"]["accelerator_telemetry"], {"mode": "disabled"})
        self.assertEqual(
            cuda["soak"]["gpu_memory_source"],
            'server_metrics:kiln_gpu_memory_bytes{kind="used"}',
        )
        self.assertEqual(
            cuda["memory"],
            {"floor_gb": 1.5, "inference_memory_fraction": 0.7},
        )
        metal = soak.effective_config(
            soak.METAL_ENDURANCE_DURATION_SECONDS,
            soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
            soak.METAL_ENDURANCE_RUNTIME,
        )
        self.assertEqual(metal["build"]["features"], "metal")
        self.assertEqual(metal["build"]["cargo_execution_mode"], "macos-contained")
        self.assertEqual(metal["server"]["max_active_requests"], 8)
        self.assertEqual(metal["server"]["max_prefill_staging_slots"], 4)
        self.assertEqual(
            metal["batching"],
            {"actor_cycle_idle_ms": 0, "prefill_admission_quantum": 4},
        )
        self.assertEqual(
            metal["soak"]["gpu_memory_scope"],
            "macos_whole_host_unified_memory",
        )
        self.assertEqual(
            metal["soak"]["gpu_memory_absolute_limit_bytes"], 15 * 1024**3
        )
        self.assertEqual(metal["soak"]["gpu_memory_poll_interval_ms"], 1000)
        self.assertEqual(
            metal["soak"]["external_yield_sync_slow_policy"],
            "record_only_failures_and_unexplained_itl_remain_fatal",
        )
        self.assertEqual(
            metal["soak"]["stabilization_memory_boundary"],
            "whole_host_unified_and_process_rss",
        )
        self.assertEqual(
            metal["memory"],
            {
                "gpu_memory_gb": 12.0,
                "floor_gb": 1.0,
                "inference_memory_fraction": 0.7,
            },
        )
        self.assertEqual(
            soak.effective_config(60.0, 123)["soak"][
                "active_gpu_peak_growth_limit_bytes"
            ],
            123,
        )

    def test_vulkan_launch_file_enforces_the_qualified_active_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "kiln.toml"
            soak.mixed.write_server_config(
                path,
                soak.VULKAN_RUNTIME.variant_id,
                root / "model",
                8420,
                root / "adapters",
                root / "snapshots",
            )
            parsed = parse_generated_toml(path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["server"]["max_decode_batch"], 2)
        self.assertEqual(parsed["batching"]["prefill_admission_quantum"], 2)
        self.assertEqual(parsed["memory"]["vulkan_buffer_pool_gb"], 3.5)

    def test_cuda_launch_file_is_portable_and_uses_the_measured_memory_envelope(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "kiln.toml"
            runtime = soak.CUDA_ENDURANCE_RUNTIME
            soak.mixed.write_server_config(
                path,
                runtime.variant_id,
                root / "model",
                8420,
                root / "adapters",
                root / "snapshots",
                rocm_graph_cache_entries=runtime.graph_cache_max,
                inference_memory_fraction=runtime.inference_memory_fraction,
                memory_floor_gb=runtime.memory_floor_gb,
            )
            parsed = parse_generated_toml(path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["server"]["serving_profile"], "stable")
        self.assertEqual(parsed["server"]["max_decode_batch"], 4)
        self.assertEqual(parsed["memory"]["inference_memory_fraction"], 0.7)
        self.assertEqual(parsed["memory"]["floor_gb"], 1.5)
        self.assertFalse(parsed["memory"]["cuda_graphs"])
        self.assertEqual(parsed["memory"]["vulkan_buffer_pool_gb"], 0.0)

    def test_metal_launch_file_uses_the_measured_unified_memory_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "kiln.toml"
            runtime = soak.METAL_ENDURANCE_RUNTIME
            soak.mixed.write_server_config(
                path,
                runtime.variant_id,
                root / "model",
                8420,
                root / "adapters",
                root / "snapshots",
                gpu_memory_gb=runtime.gpu_memory_gb,
                inference_memory_fraction=runtime.inference_memory_fraction,
                memory_floor_gb=runtime.memory_floor_gb,
            )
            parsed = parse_generated_toml(path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["server"]["serving_profile"], "stable")
        self.assertEqual(parsed["server"]["max_decode_batch"], 4)
        self.assertEqual(parsed["memory"]["gpu_memory_gb"], 12.0)
        self.assertEqual(parsed["memory"]["inference_memory_fraction"], 0.7)
        self.assertEqual(parsed["memory"]["floor_gb"], 1.0)

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

    def test_process_memory_snapshot_reads_darwin_libproc_rss(self) -> None:
        library = mock.Mock()

        def proc_pidinfo(
            _pid: int,
            _flavor: int,
            _argument: int,
            buffer: object,
            size: int,
        ) -> int:
            info = soak.ctypes.cast(
                buffer,
                soak.ctypes.POINTER(soak.DarwinProcTaskInfo),
            ).contents
            info.pti_resident_size = 12345
            return size

        library.proc_pidinfo.side_effect = proc_pidinfo
        loader = mock.Mock(return_value=library)

        snapshot = soak.process_memory_snapshot(
            42,
            platform_name="darwin",
            library_loader=loader,
        )

        self.assertEqual(snapshot.rss_bytes, 12345)
        self.assertEqual(snapshot.rss_anon_bytes, 0)
        self.assertEqual(snapshot.rss_file_bytes, 0)
        self.assertEqual(snapshot.rss_shmem_bytes, 0)
        self.assertEqual(snapshot.swap_bytes, 0)
        loader.assert_called_once_with("/usr/lib/libproc.dylib")
        self.assertEqual(library.proc_pidinfo.call_args.args[:3], (42, 4, 0))

    def test_process_memory_mapping_categories_are_closed(self) -> None:
        self.assertEqual(soak.process_memory_mapping_category(""), "anonymous")
        self.assertEqual(
            soak.process_memory_mapping_category("[anon:jemalloc]"), "anonymous"
        )
        self.assertEqual(soak.process_memory_mapping_category("[heap]"), "heap")
        self.assertEqual(soak.process_memory_mapping_category("[stack:42]"), "stack")
        self.assertEqual(
            soak.process_memory_mapping_category("/dev/dri/renderD128"), "device"
        )
        self.assertEqual(
            soak.process_memory_mapping_category("/memfd:mesa-shared (deleted)"),
            "shared_memory",
        )
        self.assertEqual(soak.process_memory_mapping_category("[vdso]"), "kernel")
        self.assertEqual(
            soak.process_memory_mapping_category("/models/model.safetensors"), "file"
        )

    def test_process_memory_mapping_snapshot_and_growth_are_exact(self) -> None:
        def mapping(
            address_range: str,
            pathname: str,
            *,
            size: int,
            rss: int,
            pss: int,
            anonymous: int,
            anonymous_huge: int,
            private_dirty: int,
            swap: int,
        ) -> str:
            suffix = f" {pathname}" if pathname else ""
            return (
                f"{address_range} rw-p 00000000 00:00 0{suffix}\n"
                f"Size: {size} kB\n"
                f"Rss: {rss} kB\n"
                f"Pss: {pss} kB\n"
                f"Private_Dirty: {private_dirty} kB\n"
                f"Anonymous: {anonymous} kB\n"
                f"AnonHugePages: {anonymous_huge} kB\n"
                f"Swap: {swap} kB\n"
                "VmFlags: rd wr mr mw me ac sd\n"
            )

        before_raw = mapping(
            "1000-2000",
            "",
            size=64,
            rss=16,
            pss=16,
            anonymous=16,
            anonymous_huge=8,
            private_dirty=12,
            swap=0,
        ) + mapping(
            "3000-4000",
            "/dev/dri/renderD128",
            size=128,
            rss=32,
            pss=30,
            anonymous=32,
            anonymous_huge=0,
            private_dirty=32,
            swap=0,
        )
        after_raw = mapping(
            "1000-2000",
            "",
            size=64,
            rss=48,
            pss=48,
            anonymous=48,
            anonymous_huge=40,
            private_dirty=44,
            swap=4,
        ) + mapping(
            "3000-4000",
            "/dev/dri/renderD128",
            size=128,
            rss=40,
            pss=38,
            anonymous=40,
            anonymous_huge=0,
            private_dirty=40,
            swap=0,
        )

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            smaps = root / "42" / "smaps"
            smaps.parent.mkdir(parents=True)
            smaps.write_text(before_raw, encoding="utf-8")
            before = soak.process_memory_mapping_snapshot(42, root)
            smaps.write_text(after_raw, encoding="utf-8")
            after = soak.process_memory_mapping_snapshot(42, root)

        self.assertEqual(len(before.mappings), 2)
        self.assertEqual(before.mappings[0].category, "anonymous")
        self.assertEqual(before.mappings[1].category, "device")
        totals = soak.process_memory_mapping_totals(after)
        self.assertEqual(totals["rss_bytes"], 88 * 1024)
        self.assertEqual(totals["anonymous_bytes"], 88 * 1024)
        self.assertEqual(totals["anonymous_rss_bytes"], 48 * 1024)
        self.assertEqual(totals["device_rss_bytes"], 40 * 1024)

        trace = soak.process_memory_mapping_trace(before, after, top_limit=1)
        self.assertEqual(trace["smaps_anonymous_delta_bytes"], 40 * 1024)
        self.assertEqual(
            trace["smaps_anonymous_huge_pages_delta_bytes"], 32 * 1024
        )
        self.assertEqual(
            trace["smaps_rss_delta_bytes_by_category"]["anonymous"], 32 * 1024
        )
        self.assertEqual(len(trace["smaps_top_rss_growth"]), 1)
        self.assertEqual(
            trace["smaps_top_rss_growth"][0]["rss_delta_bytes"], 32 * 1024
        )
        metrics = soak.process_memory_mapping_metric_values(before, after)
        self.assertEqual(
            metrics["vulkan_process_mapping_anonymous_rss_growth_bytes"],
            32 * 1024,
        )
        self.assertEqual(
            metrics["vulkan_process_mapping_device_rss_growth_bytes"], 8 * 1024
        )
        self.assertEqual(
            metrics["vulkan_process_smaps_private_dirty_growth_bytes"], 40 * 1024
        )
        self.assertEqual(
            metrics["vulkan_process_smaps_anonymous_huge_pages_growth_bytes"],
            32 * 1024,
        )

    def test_process_memory_mapping_snapshot_fails_on_incomplete_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            smaps = root / "42" / "smaps"
            smaps.parent.mkdir(parents=True)
            smaps.write_text(
                "1000-2000 rw-p 00000000 00:00 0\n"
                "Size: 4 kB\n"
                "Rss: 4 kB\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(soak.SoakError, "omitted required fields"):
                soak.process_memory_mapping_snapshot(42, root)

    def test_vulkan_pool_miss_attribution_is_closed_and_interval_exact(self) -> None:
        def pool_health(
            *,
            cache_misses: int,
            device_misses: int,
            host_misses: int,
            last_miss: dict | None,
        ) -> dict:
            return {
                "vulkan_buffer_pool": {
                    "max_retained_bytes": round(
                        soak.VULKAN_QUALIFIED_BUFFER_POOL_GB * 1024**3
                    ),
                    "bucket_count": 2,
                    "buffer_count": 3,
                    "retained_bytes": 3072,
                    "free_buffer_count": 2,
                    "free_bytes": 2048,
                    "borrowed_buffer_count": 1,
                    "borrowed_bytes": 1024,
                    "cache_hits": 100,
                    "cache_misses": cache_misses,
                    "device_local_cache_misses": device_misses,
                    "host_visible_cache_misses": host_misses,
                    "last_cache_miss": last_miss,
                    "eviction_count": 4,
                    "evicted_bytes": 1024,
                    "uncached_allocation_count": 0,
                    "uncached_allocated_bytes": 0,
                }
            }

        before_last = {
            "sequence": 9,
            "route": "host_visible",
            "requested_bytes": 1024,
            "bucket_bytes": 65_536,
            "caller_file": "crates/kiln-vulkan-kernel/src/buffer.rs",
            "caller_line": 484,
        }
        after_last = {
            "sequence": 10,
            "route": "device_local",
            "requested_bytes": 20_000_000,
            "bucket_bytes": 20_971_520,
            "caller_file": "crates/kiln-tensor/src/vulkan_storage.rs",
            "caller_line": 1234,
        }
        before = soak.vulkan_buffer_pool_snapshot(
            pool_health(
                cache_misses=9,
                device_misses=7,
                host_misses=2,
                last_miss=before_last,
            ),
            soak.VULKAN_RUNTIME,
        )
        after = soak.vulkan_buffer_pool_snapshot(
            pool_health(
                cache_misses=10,
                device_misses=8,
                host_misses=2,
                last_miss=after_last,
            ),
            soak.VULKAN_RUNTIME,
        )
        assert before is not None and after is not None
        trace = soak.vulkan_buffer_pool_miss_trace_values(before, after)
        self.assertEqual(trace["vulkan_pool_cache_miss_count"], 1)
        self.assertEqual(trace["vulkan_pool_device_local_cache_miss_count"], 1)
        self.assertEqual(trace["vulkan_pool_host_visible_cache_miss_count"], 0)
        self.assertEqual(trace["vulkan_pool_last_cache_miss"], after_last)
        metrics = soak.vulkan_buffer_pool_metric_values(before, after)
        self.assertEqual(metrics["vulkan_buffer_pool_cache_miss_count"], 1)
        self.assertEqual(
            metrics["vulkan_buffer_pool_device_local_cache_miss_count"], 1
        )
        self.assertEqual(
            metrics["vulkan_buffer_pool_host_visible_cache_miss_count"], 0
        )

        with self.assertRaisesRegex(soak.SoakError, "route miss counters"):
            soak.vulkan_buffer_pool_snapshot(
                pool_health(
                    cache_misses=10,
                    device_misses=7,
                    host_misses=2,
                    last_miss=after_last,
                ),
                soak.VULKAN_RUNTIME,
            )
        wrong_limit = pool_health(
            cache_misses=9,
            device_misses=7,
            host_misses=2,
            last_miss=before_last,
        )
        wrong_limit["vulkan_buffer_pool"]["max_retained_bytes"] = 3 * 1024**3
        with self.assertRaisesRegex(soak.SoakError, "configured cap mismatch"):
            soak.vulkan_buffer_pool_snapshot(wrong_limit, soak.VULKAN_RUNTIME)

    def test_vulkan_stabilization_rejects_allocator_churn(self) -> None:
        stable = {
            "gpu_delta": 0,
            "rss_delta": 0,
            "vulkan_live_bytes_delta": 0,
            "vulkan_allocation_count": 0,
            "vulkan_free_count": 0,
            "vulkan_pool_cache_miss_count": 0,
            "vulkan_pool_eviction_count": 0,
            "vulkan_pool_uncached_allocation_count": 0,
        }
        self.assertTrue(
            soak.stabilization_cycle_is_stable(soak.VULKAN_RUNTIME, **stable)
        )
        for field in (
            "vulkan_live_bytes_delta",
            "vulkan_allocation_count",
            "vulkan_free_count",
            "vulkan_pool_cache_miss_count",
            "vulkan_pool_eviction_count",
            "vulkan_pool_uncached_allocation_count",
        ):
            with self.subTest(field=field):
                churn = {**stable, field: 1}
                self.assertFalse(
                    soak.stabilization_cycle_is_stable(
                        soak.VULKAN_RUNTIME, **churn
                    )
                )
        self.assertFalse(
            soak.stabilization_cycle_is_stable(
                soak.VULKAN_RUNTIME,
                **{
                    **stable,
                    "gpu_delta": (
                        soak.VULKAN_RUNTIME.stabilization_gpu_delta_limit_bytes + 1
                    ),
                },
            )
        )

    def test_cuda_stabilization_compares_against_envelope_high_water(self) -> None:
        self.assertEqual(
            soak.stabilization_gpu_growth_delta(
                soak.CUDA_ENDURANCE_RUNTIME,
                current_gpu=250,
                previous_gpu=200,
                stabilization_gpu_high_water=300,
            ),
            0,
        )
        self.assertEqual(
            soak.stabilization_gpu_growth_delta(
                soak.CUDA_ENDURANCE_RUNTIME,
                current_gpu=350,
                previous_gpu=200,
                stabilization_gpu_high_water=300,
            ),
            50,
        )
        self.assertEqual(
            soak.measurement_gpu_baseline(
                soak.CUDA_ENDURANCE_RUNTIME,
                current_gpu=250,
                stabilization_gpu_high_water=300,
            ),
            300,
        )
        self.assertEqual(
            soak.stabilization_gpu_growth_delta(
                soak.VULKAN_RUNTIME,
                current_gpu=250,
                previous_gpu=200,
                stabilization_gpu_high_water=300,
            ),
            50,
        )

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

    def test_gpu_memory_sampler_reads_the_macos_unified_counter(self) -> None:
        counter = mock.Mock()
        counter.read_bytes.return_value = 987
        with mock.patch.object(
            soak, "MacOsUnifiedMemoryCounter", return_value=counter
        ) as counter_type:
            sampler = soak.GpuMemorySampler(
                8420, 42, soak.METAL_ENDURANCE_RUNTIME
            )
            sampler._sample()
            sampler.close()

        self.assertEqual(sampler.samples, [987])
        self.assertEqual(sampler.errors, [])
        counter_type.assert_called_once_with()
        counter.read_bytes.assert_called_once_with()
        counter.close.assert_called_once_with()

    def test_vulkan_buffer_snapshot_is_closed_typed_and_monotonic(self) -> None:
        before = {
            "live_device_local_buffers": 5,
            "live_device_local_bytes": 100,
            "live_host_visible_buffers": 3,
            "live_host_visible_bytes": 20,
            "peak_live_bytes": 120,
            "device_local_allocations": 8,
            "device_local_allocated_bytes": 160,
            "device_local_frees": 3,
            "device_local_freed_bytes": 60,
            "host_visible_allocations": 7,
            "host_visible_allocated_bytes": 70,
            "host_visible_frees": 4,
            "host_visible_freed_bytes": 50,
        }
        after = dict(before)
        after.update(
            live_device_local_buffers=6,
            live_device_local_bytes=130,
            peak_live_bytes=150,
            device_local_allocations=10,
            device_local_allocated_bytes=200,
            device_local_frees=4,
            device_local_freed_bytes=70,
        )
        self.assertEqual(
            soak.vulkan_buffer_snapshot(
                {"vulkan_buffers": before}, soak.VULKAN_RUNTIME
            ),
            before,
        )
        self.assertIsNone(
            soak.vulkan_buffer_snapshot({"vulkan_buffers": before}, soak.ROCM_RUNTIME)
        )
        self.assertEqual(soak.vulkan_buffer_live_bytes(before), 120)
        self.assertEqual(
            soak.vulkan_buffer_counter_delta(before, after, "allocated_bytes"), 40
        )
        self.assertEqual(
            soak.vulkan_buffer_counter_delta(before, after, "allocations"), 2
        )
        self.assertEqual(soak.vulkan_buffer_live_count(before), 8)
        self.assertEqual(
            soak.vulkan_buffer_metric_values(before, after),
            {
                "vulkan_buffer_allocated_bytes": 40,
                "vulkan_buffer_allocation_count": 2,
                "vulkan_buffer_free_count": 1,
                "vulkan_buffer_freed_bytes": 10,
                "vulkan_buffer_live_bytes_end": 150,
                "vulkan_buffer_live_bytes_growth": 30,
                "vulkan_buffer_live_bytes_start": 120,
                "vulkan_buffer_peak_live_bytes": 150,
            },
        )
        self.assertEqual(soak.vulkan_buffer_accounting_failures(before, after), [])
        inconsistent = dict(after)
        inconsistent["device_local_freed_bytes"] = 69
        self.assertRegex(
            soak.vulkan_buffer_accounting_failures(before, inconsistent)[0],
            "byte accounting is inconsistent",
        )
        with self.assertRaisesRegex(soak.SoakError, "field mismatch"):
            soak.vulkan_buffer_snapshot(
                {"vulkan_buffers": {**before, "future": 1}}, soak.VULKAN_RUNTIME
            )
        regressed = dict(after)
        regressed["device_local_allocations"] = 1
        with self.assertRaisesRegex(soak.SoakError, "regressed"):
            soak.vulkan_buffer_counter_delta(after, regressed, "allocations")

    def test_batched_state_cache_snapshot_is_closed_typed_and_reconciled(
        self,
    ) -> None:
        snapshot = soak.batched_state_cache_snapshot(
            batched_state_debug(), soak.VULKAN_RUNTIME
        )
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(set(snapshot), set(soak.BATCHED_STATE_CACHE_FIELDS))
        self.assertTrue(snapshot["entry_present"])
        self.assertEqual(snapshot["capacity_rows"], 8)
        self.assertIsNone(
            soak.batched_state_cache_snapshot(
                batched_state_debug(), soak.ROCM_RUNTIME
            )
        )

        malformed = batched_state_debug()
        malformed["caches"]["batched_recurrent_state"]["future"] = 0
        with self.assertRaisesRegex(soak.SoakError, "field mismatch"):
            soak.batched_state_cache_snapshot(malformed, soak.VULKAN_RUNTIME)
        with self.assertRaisesRegex(soak.SoakError, "must be boolean"):
            soak.batched_state_cache_snapshot(
                batched_state_debug(entry_present=1), soak.VULKAN_RUNTIME
            )
        with self.assertRaisesRegex(soak.SoakError, "logical rows exceed capacity"):
            soak.batched_state_cache_snapshot(
                batched_state_debug(capacity_rows=7), soak.VULKAN_RUNTIME
            )
        with self.assertRaisesRegex(soak.SoakError, "carries parked ownership"):
            soak.batched_state_cache_snapshot(
                batched_state_debug(entry_present=False), soak.VULKAN_RUNTIME
            )
        with self.assertRaisesRegex(soak.SoakError, "exceed the process peak"):
            soak.batched_state_cache_snapshot(
                batched_state_debug(active_leases=2), soak.VULKAN_RUNTIME
            )

    def test_resident_recurrent_state_snapshot_is_closed_and_drain_gated(self) -> None:
        snapshot = soak.resident_recurrent_state_snapshot(
            batched_state_debug(), soak.VULKAN_RUNTIME
        )
        self.assertEqual(
            snapshot,
            {"entry_count": 0, "buffer_bytes": 0, "allocation_bytes": 0},
        )
        self.assertEqual(
            soak.resident_recurrent_state_drain_failures(snapshot, "test drain"), []
        )
        self.assertIsNone(
            soak.resident_recurrent_state_snapshot(
                batched_state_debug(), soak.ROCM_RUNTIME
            )
        )

        retained = soak.resident_recurrent_state_snapshot(
            batched_state_debug(
                resident_overrides={
                    "entry_count": 2,
                    "buffer_bytes": 1024,
                    "allocation_bytes": 1280,
                }
            ),
            soak.VULKAN_RUNTIME,
        )
        self.assertRegex(
            soak.resident_recurrent_state_drain_failures(retained, "test drain")[0],
            "entries=2, buffer_bytes=1024, allocation_bytes=1280",
        )
        self.assertEqual(
            soak.resident_recurrent_state_metric_values(retained),
            {
                "resident_recurrent_state_entries_end": 2,
                "resident_recurrent_state_buffer_bytes_end": 1024,
                "resident_recurrent_state_allocation_bytes_end": 1280,
            },
        )

        malformed = batched_state_debug()
        malformed["caches"]["resident_recurrent_state"]["future"] = 0
        with self.assertRaisesRegex(soak.SoakError, "field mismatch"):
            soak.resident_recurrent_state_snapshot(malformed, soak.VULKAN_RUNTIME)
        with self.assertRaisesRegex(soak.SoakError, "below buffer bytes"):
            soak.resident_recurrent_state_snapshot(
                batched_state_debug(
                    resident_overrides={
                        "entry_count": 1,
                        "buffer_bytes": 1024,
                        "allocation_bytes": 512,
                    }
                ),
                soak.VULKAN_RUNTIME,
            )
        with self.assertRaisesRegex(soak.SoakError, "retains bytes"):
            soak.resident_recurrent_state_snapshot(
                batched_state_debug(
                    resident_overrides={
                        "entry_count": 0,
                        "buffer_bytes": 1024,
                        "allocation_bytes": 1024,
                    }
                ),
                soak.VULKAN_RUNTIME,
            )

    def test_batched_state_cache_deltas_retain_lifecycle_attribution(self) -> None:
        before = soak.batched_state_cache_snapshot(
            batched_state_debug(), soak.VULKAN_RUNTIME
        )
        after = soak.batched_state_cache_snapshot(
            batched_state_debug(
                capacity_rows=16,
                logical_rows=6,
                max_active_leases=2,
                take_hit_count=7,
                take_miss_count=3,
                take_miss_while_leased_count=2,
                resident_prefix_view_count=4,
                resident_prefix_snapshot_suppression_count=3,
                fresh_assembly_count=1,
                park_count=8,
                park_replacement_eviction_count=2,
            ),
            soak.VULKAN_RUNTIME,
        )
        assert before is not None and after is not None

        trace = soak.batched_state_cache_trace_values(before, after)
        self.assertEqual(trace["batched_state_cache_capacity_rows"], 16)
        self.assertEqual(trace["batched_state_cache_take_hit_count_delta"], 7)
        self.assertEqual(
            trace["batched_state_cache_park_replacement_eviction_count_delta"], 2
        )
        self.assertEqual(
            trace[
                "batched_state_cache_resident_prefix_snapshot_suppression_count_delta"
            ],
            3,
        )

        metrics = soak.batched_state_cache_metric_values(before, after)
        expected = {
            name
            for name in soak.metric_definitions(soak.VULKAN_RUNTIME)
            if name.startswith("batched_state_cache_")
        }
        self.assertEqual(set(metrics), expected)
        self.assertEqual(metrics["batched_state_cache_max_active_leases"], 2)
        self.assertEqual(metrics["batched_state_cache_take_miss_count"], 3)
        self.assertEqual(
            metrics[
                "batched_state_cache_resident_prefix_snapshot_suppression_count"
            ],
            3,
        )
        self.assertEqual(
            metrics["batched_state_cache_park_replacement_eviction_count"], 2
        )

        regressed = dict(after)
        regressed["take_hit_count"] = 6
        with self.assertRaisesRegex(soak.SoakError, "regressed"):
            soak.batched_state_cache_metric_values(after, regressed)

    def test_resident_prefill_metrics_prove_multi_row_execution_and_drain(
        self,
    ) -> None:
        before = {
            "resident_prefill_enabled": True,
            "active_resident_prefill": 0,
            "total_resident_prefill_attempts": 2,
            "total_resident_prefill_completed_rows": 1,
            "total_resident_prefill_forwards": 2,
            "total_resident_prefill_initial_declines": 0,
            "max_resident_prefill_batch_size": 2,
            "total_resident_prefill_route_failures": 0,
            "total_resident_prefill_rows": 4,
        }
        after = {
            "resident_prefill_enabled": True,
            "active_resident_prefill": 0,
            "total_resident_prefill_attempts": 5,
            "total_resident_prefill_completed_rows": 3,
            "total_resident_prefill_forwards": 5,
            "total_resident_prefill_initial_declines": 0,
            "max_resident_prefill_batch_size": 4,
            "total_resident_prefill_route_failures": 0,
            "total_resident_prefill_rows": 10,
        }
        values = soak.resident_prefill_metric_values(before, after)
        self.assertEqual(
            values,
            {
                "resident_prefill_enabled": 1,
                "resident_prefill_active_rows_end": 0,
                "resident_prefill_attempt_count": 3,
                "resident_prefill_completed_row_count": 2,
                "resident_prefill_forward_count": 3,
                "resident_prefill_initial_decline_count": 0,
                "resident_prefill_max_batch_size": 4,
                "resident_prefill_route_failure_count": 0,
                "resident_prefill_row_count": 6,
            },
        )
        self.assertEqual(
            soak.resident_prefill_contract_failures(
                values, max_configured_rows=8
            ),
            [],
        )
        self.assertEqual(
            soak.stabilization_resident_prefill_metric_values(before, after),
            {f"stabilization_{name}": value for name, value in values.items()},
        )
        self.assertEqual(
            soak.partial_stabilization_resident_prefill_metric_values(
                soak.ROCM_RUNTIME, before, after
            ),
            {},
        )
        self.assertEqual(
            soak.partial_stabilization_resident_prefill_metric_values(
                soak.VULKAN_RUNTIME, before, after
            ),
            {f"stabilization_{name}": value for name, value in values.items()},
        )
        self.assertEqual(
            soak.partial_stabilization_resident_prefill_metric_values(
                soak.VULKAN_RUNTIME, before, None
            ),
            {},
        )

        single_row_only = dict(values)
        single_row_only["resident_prefill_row_count"] = single_row_only[
            "resident_prefill_forward_count"
        ]
        self.assertTrue(
            any(
                "did not prove a measured multi-row forward" in failure
                for failure in soak.resident_prefill_contract_failures(
                    single_row_only, max_configured_rows=8
                )
            )
        )

        broken = dict(values)
        broken.update(
            resident_prefill_active_rows_end=1,
            resident_prefill_attempt_count=4,
            resident_prefill_completed_row_count=7,
            resident_prefill_initial_decline_count=1,
            resident_prefill_max_batch_size=9,
            resident_prefill_route_failure_count=1,
        )
        failures = soak.resident_prefill_contract_failures(
            broken, max_configured_rows=8
        )
        self.assertTrue(any("active_rows_end=1" in failure for failure in failures))
        self.assertTrue(any("initial_decline_count=1" in failure for failure in failures))
        self.assertTrue(any("route_failure_count=1" in failure for failure in failures))
        self.assertTrue(any("completed rows" in failure for failure in failures))
        self.assertTrue(any("configured concurrency" in failure for failure in failures))

        regressed = dict(after)
        regressed["max_resident_prefill_batch_size"] = 1
        with self.assertRaisesRegex(soak.SoakError, "maximum batch size regressed"):
            soak.resident_prefill_metric_values(after, regressed)

    def test_disabled_resident_prefill_requires_zero_route_activity(self) -> None:
        before = {
            "resident_prefill_enabled": False,
            "active_resident_prefill": 0,
            "total_resident_prefill_attempts": 0,
            "total_resident_prefill_completed_rows": 0,
            "total_resident_prefill_forwards": 0,
            "total_resident_prefill_initial_declines": 0,
            "max_resident_prefill_batch_size": 0,
            "total_resident_prefill_route_failures": 0,
            "total_resident_prefill_rows": 0,
        }
        values = soak.resident_prefill_metric_values(before, dict(before))
        self.assertEqual(values["resident_prefill_enabled"], 0)
        self.assertEqual(
            soak.resident_prefill_contract_failures(values, max_configured_rows=8),
            [],
        )

        active = dict(values)
        active["resident_prefill_attempt_count"] = 1
        self.assertTrue(
            any(
                "while resident prefill is disabled" in failure
                for failure in soak.resident_prefill_contract_failures(
                    active, max_configured_rows=8
                )
            )
        )

    def test_disabled_prefix_cache_requires_zero_activity_and_residency(self) -> None:
        snapshot = {
            "lookup_hits": 0,
            "lookup_misses": 0,
            "hit_tokens": 0,
            "hit_blocks": 0,
            "cached_blocks": 0,
            "max_blocks": 0,
            "cached_entries": 0,
            "max_entries": 0,
            "cached_state_bytes": 0,
            "max_state_bytes": 0,
            "active_leases": 0,
            "pending_release_entries": 0,
        }
        self.assertEqual(
            soak.disabled_prefix_cache_failures(snapshot, phase="test"), []
        )
        self.assertEqual(
            soak.prefix_cache_capability_value(
                {"prefix_cache_enabled": False},
                {"prefix_cache_enabled": False},
            ),
            0,
        )

        active = dict(snapshot)
        active["lookup_misses"] = 1
        self.assertEqual(
            soak.disabled_prefix_cache_failures(active, phase="test"),
            ["test prefix-cache lookup_misses=1 while disabled"],
        )
        with self.assertRaisesRegex(soak.SoakError, "changed during the run"):
            soak.prefix_cache_capability_value(
                {"prefix_cache_enabled": False},
                {"prefix_cache_enabled": True},
            )

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
                    "active_resident_prefill": 0,
                    "active_staged_requests": 0,
                    "queue_depth": 0,
                    "actor_cycle_idle_active": False,
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
                wave=1,
                base_seed=7,
                deadline=time.monotonic() + 1.0,
                runtime=soak.VULKAN_RUNTIME,
            )
        self.assertEqual(run.call_count, 4)
        self.assertTrue(
            all(call.kwargs["max_tokens"] == 16 for call in run.call_args_list)
        )
        self.assertTrue(
            all(
                call.kwargs["request_timeout_seconds"] == 600.0
                for call in run.call_args_list
            )
        )
        self.assertEqual(
            [call.kwargs["prompt_words"] for call in run.call_args_list],
            [16, 16, 16, 16],
        )

        with mock.patch.object(soak.mixed, "run_stream", return_value=object()) as run:
            soak.run_wave(
                8420,
                wave=3,
                base_seed=7,
                deadline=time.monotonic() + 1.0,
                runtime=soak.VULKAN_RUNTIME,
            )
        self.assertEqual(
            [call.kwargs["prompt_words"] for call in run.call_args_list],
            [32, 32, 32, 32],
        )

    def test_wave_cleanup_preserves_the_primary_deadline_failure(self) -> None:
        entered = threading.Event()
        finished = threading.Event()

        def blocked_stream(*_args: object, **kwargs: object) -> object:
            entered.set()
            try:
                abort_event = kwargs["abort_event"]
                assert isinstance(abort_event, threading.Event)
                abort_event.wait(timeout=2.0)
                return object()
            finally:
                finished.set()

        def expired(*_args: object, **_kwargs: object) -> float:
            self.assertTrue(entered.wait(timeout=1.0))
            raise TimeoutError("soak stabilize wave exceeded the setup deadline")

        evidence = soak.RequestWorkerEvidence()
        with (
            mock.patch.object(soak.mixed, "run_stream", side_effect=blocked_stream),
            mock.patch.object(soak.mixed, "remaining_until", side_effect=expired),
        ):
            with self.assertRaisesRegex(
                TimeoutError, "soak stabilize wave exceeded the setup deadline"
            ):
                soak.run_wave(
                    8420,
                    wave=0,
                    base_seed=7,
                    deadline=time.monotonic() - 1.0,
                    phase="stabilize",
                    worker_evidence=evidence,
                )
        self.assertTrue(finished.wait(timeout=1.0))
        self.assertEqual(evidence.peak_residue_count, 0)

    def test_wave_cleanup_retains_worker_residue_evidence(self) -> None:
        release = threading.Event()

        def blocked_stream(*_args: object, **_kwargs: object) -> object:
            release.wait(timeout=2.0)
            return object()

        evidence = soak.RequestWorkerEvidence()
        with (
            mock.patch.object(soak.mixed, "run_stream", side_effect=blocked_stream),
            mock.patch.object(
                soak.mixed,
                "remaining_until",
                side_effect=TimeoutError("deadline"),
            ),
            mock.patch.object(soak, "REQUEST_WORKER_CLEANUP_TIMEOUT_SECONDS", 0.0),
        ):
            try:
                with self.assertRaisesRegex(
                    soak.SoakError, "1 request workers survived wave cleanup"
                ):
                    soak.run_wave(
                        8420,
                        wave=0,
                        base_seed=7,
                        deadline=time.monotonic() - 1.0,
                        worker_evidence=evidence,
                    )
            finally:
                release.set()
        self.assertEqual(evidence.peak_residue_count, 1)

    def test_cancellation_markers_are_unique_and_confirmation_is_required(self) -> None:
        def result(*_args: object, **kwargs: object) -> mock.Mock:
            return mock.Mock(
                cancelled=True,
                semantic_times=[0.0] * soak.mixed.CANCELLATION_AFTER_DELTAS,
                semantic_deltas=[
                    {"choices": [{"delta": {"content": fragment}}]}
                    for fragment in ("000", "000 ", "000", "001")
                ],
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

        def corrupt_result(*_args: object, **kwargs: object) -> mock.Mock:
            value = result(*_args, **kwargs)
            value.semantic_deltas[-1] = {
                "choices": [{"delta": {"content": "000"}}]
            }
            return value

        with (
            mock.patch.object(
                soak.mixed, "run_stream", side_effect=corrupt_result
            ),
            mock.patch.object(
                soak.mixed,
                "wait_for_cancellation_and_drain",
                return_value=(True, {}),
            ),
        ):
            failure = soak.run_cancellation(
                8420,
                wave=4,
                base_seed=7,
                phase="measured",
                deadline=time.monotonic() + 1.0,
            )
        self.assertIn("failed ascending_zero_padded_integers_prefix_v1", failure)
        self.assertIn("response_text=", failure)

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
            case["timeout_seconds"],
            soak.qualification_case_timeout_seconds(
                soak.QUALIFICATION_DURATION_SECONDS, soak.ROCM_RUNTIME
            ),
        )
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(soak.metric_definitions(soak.ROCM_RUNTIME)),
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
            variant["effective_config"]["server"][
                "max_prefill_tokens_per_cycle"
            ],
            soak.VULKAN_MAX_PREFILL_TOKENS_PER_CYCLE,
        )
        self.assertEqual(
            variant["effective_config"]["model"],
            {
                "checkpoint_read_mib_per_second": 256,
                "accelerator_weight_upload_mib_per_second": 256,
                "vulkan_decode_weight_prewarm": True,
                "vulkan_decode_weight_prewarm_mib_per_second": 256,
            },
        )
        self.assertEqual(
            soak.mixed.VARIANT_CONFIGS["stable"]["server"][
                "max_prefill_tokens_per_cycle"
            ],
            soak.mixed.MAX_PREFILL_TOKENS_PER_CYCLE,
        )
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
        self.assertEqual(
            case["timeout_seconds"],
            soak.qualification_case_timeout_seconds(
                soak.QUALIFICATION_DURATION_SECONDS, soak.VULKAN_RUNTIME
            ),
        )

    def test_checked_in_rocm_endurance_workload_matches_driver_contract(self) -> None:
        path = ROOT / "qualification/workloads/serving-rocm-endurance-v1.json"
        workload = json.loads(path.read_text())
        self.assertEqual(workload["workload_id"], "serving-rocm-endurance-v1")
        self.assertEqual(workload["kind"], "soak")
        self.assertIsNone(workload["comparison_policy"])
        self.assertEqual(len(workload["variants"]), 1)
        variant = workload["variants"][0]
        self.assertEqual(variant["id"], soak.ROCM_ENDURANCE_RUNTIME.variant_id)
        self.assertEqual(variant["backend"], "rocm")
        self.assertEqual(
            variant["effective_config"],
            soak.effective_config(
                soak.ROCM_ENDURANCE_DURATION_SECONDS,
                soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
                soak.ROCM_ENDURANCE_RUNTIME,
            ),
        )
        self.assertEqual(len(variant["cases"]), 1)
        case = variant["cases"][0]
        self.assertEqual(case["id"], soak.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(soak.metric_definitions(soak.ROCM_ENDURANCE_RUNTIME)),
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
                "86400",
                "--memory-growth-limit-bytes",
                str(soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES),
            ],
        )
        self.assertEqual(
            case["timeout_seconds"],
            soak.qualification_case_timeout_seconds(
                soak.ROCM_ENDURANCE_DURATION_SECONDS, soak.ROCM_ENDURANCE_RUNTIME
            ),
        )

    def test_checked_in_vulkan_endurance_workload_matches_driver_contract(self) -> None:
        path = ROOT / "qualification/workloads/serving-vulkan-endurance-v1.json"
        workload = json.loads(path.read_text())
        self.assertEqual(workload["workload_id"], "serving-vulkan-endurance-v1")
        self.assertEqual(workload["kind"], "soak")
        self.assertIsNone(workload["comparison_policy"])
        self.assertEqual(len(workload["variants"]), 1)
        variant = workload["variants"][0]
        self.assertEqual(variant["id"], soak.VULKAN_ENDURANCE_RUNTIME.variant_id)
        self.assertEqual(variant["backend"], "vulkan")
        self.assertEqual(
            variant["effective_config"],
            soak.effective_config(
                soak.VULKAN_ENDURANCE_DURATION_SECONDS,
                soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
                soak.VULKAN_ENDURANCE_RUNTIME,
            ),
        )
        self.assertEqual(len(variant["cases"]), 1)
        case = variant["cases"][0]
        self.assertEqual(case["id"], soak.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(soak.metric_definitions(soak.VULKAN_ENDURANCE_RUNTIME)),
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
                "28800",
                "--memory-growth-limit-bytes",
                str(soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES),
            ],
        )
        self.assertEqual(
            case["timeout_seconds"],
            soak.qualification_case_timeout_seconds(
                soak.VULKAN_ENDURANCE_DURATION_SECONDS,
                soak.VULKAN_ENDURANCE_RUNTIME,
            ),
        )

    def test_checked_in_cuda_endurance_workload_matches_driver_contract(self) -> None:
        path = ROOT / "qualification/workloads/serving-cuda-endurance-v1.json"
        workload = json.loads(path.read_text())
        self.assertEqual(workload["workload_id"], "serving-cuda-endurance-v1")
        self.assertEqual(workload["kind"], "soak")
        self.assertIsNone(workload["comparison_policy"])
        self.assertEqual(len(workload["variants"]), 1)
        variant = workload["variants"][0]
        self.assertEqual(variant["id"], soak.CUDA_ENDURANCE_RUNTIME.variant_id)
        self.assertEqual(variant["backend"], "cuda")
        self.assertEqual(
            variant["effective_config"],
            soak.effective_config(
                soak.CUDA_ENDURANCE_DURATION_SECONDS,
                soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
                soak.CUDA_ENDURANCE_RUNTIME,
            ),
        )
        self.assertEqual(len(variant["cases"]), 1)
        case = variant["cases"][0]
        self.assertEqual(case["id"], soak.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(soak.metric_definitions(soak.CUDA_ENDURANCE_RUNTIME)),
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
                "28800",
                "--memory-growth-limit-bytes",
                str(soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES),
            ],
        )
        self.assertEqual(
            case["timeout_seconds"],
            soak.qualification_case_timeout_seconds(
                soak.CUDA_ENDURANCE_DURATION_SECONDS,
                soak.CUDA_ENDURANCE_RUNTIME,
            ),
        )

    def test_checked_in_metal_endurance_workload_matches_driver_contract(self) -> None:
        path = ROOT / "qualification/workloads/serving-metal-endurance-v1.json"
        workload = json.loads(path.read_text())
        self.assertEqual(workload["workload_id"], "serving-metal-endurance-v1")
        self.assertEqual(workload["kind"], "soak")
        self.assertIsNone(workload["comparison_policy"])
        self.assertEqual(len(workload["variants"]), 1)
        variant = workload["variants"][0]
        self.assertEqual(variant["id"], soak.METAL_ENDURANCE_RUNTIME.variant_id)
        self.assertEqual(variant["backend"], "metal")
        self.assertEqual(
            variant["effective_config"],
            soak.effective_config(
                soak.METAL_ENDURANCE_DURATION_SECONDS,
                soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
                soak.METAL_ENDURANCE_RUNTIME,
            ),
        )
        self.assertEqual(len(variant["cases"]), 1)
        case = variant["cases"][0]
        self.assertEqual(case["id"], soak.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(soak.metric_definitions(soak.METAL_ENDURANCE_RUNTIME)),
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
                "28800",
                "--memory-growth-limit-bytes",
                str(soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES),
            ],
        )
        self.assertEqual(
            case["timeout_seconds"],
            soak.qualification_case_timeout_seconds(
                soak.METAL_ENDURANCE_DURATION_SECONDS,
                soak.METAL_ENDURANCE_RUNTIME,
            ),
        )

    def test_checked_in_metal_development_workload_matches_driver_contract(
        self,
    ) -> None:
        path = (
            ROOT
            / "qualification/workloads/serving-metal-development-soak-v1.json"
        )
        workload = json.loads(path.read_text())
        self.assertEqual(
            workload["workload_id"], "serving-metal-development-soak-v1"
        )
        variant = workload["variants"][0]
        self.assertEqual(variant["id"], soak.METAL_ENDURANCE_RUNTIME.variant_id)
        self.assertEqual(
            variant["effective_config"],
            soak.effective_config(
                60.0,
                soak.DEFAULT_MEMORY_GROWTH_LIMIT_BYTES,
                soak.METAL_ENDURANCE_RUNTIME,
            ),
        )
        case = variant["cases"][0]
        self.assertEqual(
            case["command"][-3:],
            ["60", "--memory-growth-limit-bytes", "536870912"],
        )
        self.assertEqual(
            case["timeout_seconds"],
            soak.qualification_case_timeout_seconds(
                60.0, soak.METAL_ENDURANCE_RUNTIME
            ),
        )

    def test_phase_deadlines_and_case_timeout_are_independent(self) -> None:
        started = 100.0
        self.assertEqual(
            soak.build_phase_deadline(started, soak.VULKAN_RUNTIME), 1000.0
        )
        self.assertEqual(
            soak.setup_phase_deadline(started, soak.VULKAN_RUNTIME), 1900.0
        )
        self.assertEqual(
            soak.measurement_phase_deadline(
                started, soak.QUALIFICATION_DURATION_SECONDS, soak.VULKAN_RUNTIME
            ),
            2500.0,
        )
        self.assertEqual(
            soak.qualification_case_timeout_seconds(
                soak.QUALIFICATION_DURATION_SECONDS, soak.VULKAN_RUNTIME
            ),
            5280,
        )
        self.assertEqual(
            soak.qualification_case_timeout_seconds(
                soak.QUALIFICATION_DURATION_SECONDS, soak.ROCM_RUNTIME
            ),
            4200,
        )
        self.assertEqual(soak.phase_elapsed_seconds(None, 500.0), 0.0)
        self.assertEqual(soak.phase_elapsed_seconds(475.0, 500.0), 25.0)
        self.assertEqual(soak.phase_elapsed_seconds(501.0, 500.0), 0.0)

    def test_runtime_setup_clock_starts_after_source_build(self) -> None:
        binary = ROOT / "target/release/kiln"
        with (
            mock.patch.object(soak.time, "monotonic", side_effect=[100.0, 700.0]),
            mock.patch.object(
                soak.mixed,
                "build_binary",
                return_value=(binary, "sha256:" + "a" * 64, 600.0),
            ) as build_binary,
        ):
            result = soak.build_binary_for_soak(soak.VULKAN_RUNTIME)

        build_binary.assert_called_once_with(1000.0, soak.VULKAN_RUNTIME.build_spec)
        self.assertEqual(result[3], 700.0)
        self.assertEqual(result[4], 2500.0)

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
        self.assertEqual(
            killpg.call_args_list,
            [mock.call(1234, signal.SIGTERM)],
        )
        self.assertIn("below", guard.trip_reason or "")
        self.assertEqual(guard.metric_values()["host_memory_guard_trip_count"], 1)
        self.assertEqual(guard.metric_values()["host_swap_growth_bytes"], 150)
        self.assertEqual(
            guard.metric_values()["host_mem_available_min_bytes"], 7 * 1024**3
        )

    def test_disabled_accelerator_telemetry_has_no_backend_probe_or_metrics(
        self,
    ) -> None:
        sampler = soak.AcceleratorTelemetrySampler(enabled=False, required=False)
        with mock.patch.object(
            soak, "resolve_amd_accelerator_telemetry_paths"
        ) as resolve:
            sampler.start()
            sampler.close()
        resolve.assert_not_called()
        self.assertEqual(sampler.metric_values_since(None), {})
        self.assertEqual(sampler.errors, [])

    def test_accelerator_telemetry_resolves_and_aggregates_amd_sysfs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = write_amd_telemetry_fixture(root)
            resolved = soak.resolve_amd_accelerator_telemetry_paths(root)
            self.assertEqual(resolved.drm_device, root / "card1/device")
            self.assertEqual(resolved.advertised_max_sclk_hz, 2_900_000_000)

            sampler = soak.AcceleratorTelemetrySampler(required=True, drm_root=root)
            sampler.paths = resolved
            with mock.patch.object(soak.time, "monotonic", return_value=10.0):
                sampler._sample()
            paths["busy"].write_text("90\n", encoding="utf-8")
            paths["sclk"].write_text("1200000000\n", encoding="utf-8")
            paths["power"].write_text("55000000\n", encoding="utf-8")
            with mock.patch.object(soak.time, "monotonic", return_value=20.0):
                sampler._sample()

        self.assertEqual(
            sampler.metric_values_since(5.0),
            {
                "accelerator_gpu_busy_percent_p50": 85.0,
                "accelerator_gpu_busy_percent_peak": 90,
                "accelerator_power_active_p50_microwatts": 50_000_000.0,
                "accelerator_power_active_peak_microwatts": 55_000_000,
                "accelerator_sclk_active_below_half_max_count": 1,
                "accelerator_sclk_active_max_hz": 2_400_000_000,
                "accelerator_sclk_active_min_hz": 1_200_000_000,
                "accelerator_sclk_active_p50_hz": 1_800_000_000.0,
                "accelerator_sclk_advertised_max_hz": 2_900_000_000,
                "accelerator_telemetry_active_sample_count": 2,
                "accelerator_telemetry_available": 1,
                "accelerator_telemetry_error_count": 0,
                "accelerator_telemetry_sample_count": 2,
            },
        )
        premeasurement = sampler.metric_values_since(math.inf)
        self.assertEqual(premeasurement["accelerator_telemetry_available"], 1)
        self.assertEqual(premeasurement["accelerator_telemetry_sample_count"], 0)
        self.assertEqual(
            premeasurement["accelerator_telemetry_active_sample_count"], 0
        )
        self.assertEqual(
            premeasurement["accelerator_sclk_advertised_max_hz"],
            2_900_000_000,
        )

    def test_accelerator_telemetry_required_and_optional_unavailable_modes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            required = soak.AcceleratorTelemetrySampler(
                required=True, drm_root=root
            )
            optional = soak.AcceleratorTelemetrySampler(
                required=False, drm_root=root
            )
            with mock.patch.object(soak.mixed, "trace"):
                required.start()
                optional.start()
            required.close()
            optional.close()

        self.assertEqual(len(required.errors), 1)
        self.assertIn("resolved 0 DRM devices", required.errors[0])
        self.assertEqual(optional.errors, [])
        self.assertIsNotNone(optional.unavailable_reason)
        self.assertEqual(
            optional.metric_values_since(None)["accelerator_telemetry_available"],
            0,
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_amd_telemetry_fixture(root)
            second = root / "card2/device"
            second.mkdir(parents=True)
            (second / "vendor").write_text("0x1002\n", encoding="utf-8")
            ambiguous = soak.AcceleratorTelemetrySampler(
                required=False, drm_root=root
            )
            with mock.patch.object(soak.mixed, "trace"):
                ambiguous.start()
            ambiguous.close()

        self.assertEqual(len(ambiguous.errors), 1)
        self.assertIn("resolved 2 DRM devices", ambiguous.errors[0])






















    def test_metric_contract_is_closed_sorted_and_finite(self) -> None:
        values = {name: 0 for name in soak.metric_definitions(soak.ROCM_RUNTIME)}
        metrics = soak.metrics_from_values(values)
        self.assertEqual(
            [metric["name"] for metric in metrics], sorted(values)
        )
        self.assertEqual(
            set(soak.HOST_SAFETY_METRIC_DEFINITIONS)
            | set(soak.ACCELERATOR_TELEMETRY_METRIC_DEFINITIONS),
            set(soak.ROCM_METRIC_DEFINITIONS) - set(soak.METRIC_DEFINITIONS),
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
        cuda_values = {
            name: 0 for name in soak.metric_definitions(soak.CUDA_ENDURANCE_RUNTIME)
        }
        self.assertEqual(
            set(cuda_values),
            set(soak.METRIC_DEFINITIONS) | set(soak.HOST_SAFETY_METRIC_DEFINITIONS),
        )
        self.assertTrue(
            set(cuda_values).isdisjoint(soak.ACCELERATOR_TELEMETRY_METRIC_DEFINITIONS)
        )
        cuda_metrics = soak.metrics_from_values(
            cuda_values, soak.CUDA_ENDURANCE_RUNTIME
        )
        self.assertEqual(
            [metric["name"] for metric in cuda_metrics], sorted(cuda_values)
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
