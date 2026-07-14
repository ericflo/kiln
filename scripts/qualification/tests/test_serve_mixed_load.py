from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = QUALIFICATION_DIR.parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qualification_serve_mixed_load", QUALIFICATION_DIR / "serve_mixed_load.py"
)
assert SPEC is not None and SPEC.loader is not None
serve = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = serve
SPEC.loader.exec_module(serve)


def parse_generated_toml(source: str) -> dict[str, dict[str, object]]:
    """Parse the closed JSON-scalar TOML subset emitted by the test target."""
    parsed: dict[str, dict[str, object]] = {}
    section: dict[str, object] | None = None
    for line_number, raw in enumerate(source.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            if not name or name in parsed:
                raise AssertionError(f"invalid generated TOML section on line {line_number}")
            section = {}
            parsed[name] = section
            continue
        if section is None or "=" not in line:
            raise AssertionError(f"invalid generated TOML scalar on line {line_number}")
        key, raw_value = (part.strip() for part in line.split("=", 1))
        if not key or key in section:
            raise AssertionError(f"duplicate generated TOML key on line {line_number}")
        section[key] = json.loads(raw_value)
    return parsed


def http_fixture() -> dict:
    effective = serve.HTTP_SEND_BUFFER_BYTES
    raw = effective * 2 if sys.platform.startswith("linux") else effective
    return {
        "send_buffer_requested_bytes": serve.HTTP_SEND_BUFFER_BYTES,
        "send_buffer_kernel_readback_bytes": raw,
        "send_buffer_effective_bytes": effective,
    }


def health_fixture(
    *,
    kv_autoscale: bool,
    rocm_graphs: bool,
    serving_profile: str = "experimental",
    kv_autoscale_requested: bool | None = None,
    rocm_graphs_requested: bool | None = None,
    memory_reclaim_requested_mode: str = "off",
    memory_reclaim_mode: str = "off",
) -> dict:
    kv_autoscale_requested = (
        kv_autoscale
        if kv_autoscale_requested is None
        else kv_autoscale_requested
    )
    rocm_graphs_requested = (
        rocm_graphs if rocm_graphs_requested is None else rocm_graphs_requested
    )
    graph = {
        "requested": rocm_graphs,
        "capture_requested": rocm_graphs,
        "enabled": rocm_graphs,
        "capture_enabled": rocm_graphs,
        "state": "enabled" if rocm_graphs else "disabled",
        "unavailable_reason": None,
        "phase_telemetry_available": True,
        "phase_telemetry_unavailable_reason": None,
        "current_phase": None,
        "current_phase_elapsed_micros": 0,
        "capture_attempts": 0,
        "capture_successes": 0,
        "capture_deferrals": 0,
        "capture_failures": 0,
        "replay_attempts": 0,
        "replay_successes": 0,
        "replay_failures": 0,
        "failures": 0,
        "decode_owner_release_count": 0,
        "decode_owner_graph_release_count": 0,
        "graph_slot_create_count": 0,
        "graph_slot_reuse_count": 0,
        "cache_admission_successes": 0,
        "cache_evictions": 0,
        "cache_evicted_bytes": 0,
        "budget_evictions": 0,
        "pressure_evictions": 0,
        "invalidation_evictions": 0,
        "recovery_evictions": 0,
        "entry_capacity_rejections": 0,
        "byte_budget_rejections": 0,
        "accounting_incomplete_rejections": 0,
        "pre_capture_entry_capacity_skips": 0,
        "pre_capture_byte_budget_skips": 0,
        "pre_capture_accounting_incomplete_skips": 0,
        "pre_capture_memory_reservation_denied_skips": 0,
        "memory_governor_selector_mismatch_skips": 0,
        "max_cached_graphs": 8,
        "max_retained_bytes": 1 << 30,
        "captured_graph_count": 0,
        "graph_slot_count": 0,
        "active_graph_slot_count": 0,
        "idle_graph_slot_count": 0,
        "tracked_decode_owner_count": 0,
        "retained_stable_io_bytes": 0,
        "retained_capture_arena_bytes": 0,
        "retained_blaslt_workspace_bytes": 0,
        "retained_slot_state_bytes": 0,
        "retained_bytes": 0,
        "peak_retained_bytes": 0,
        "last_transient_candidate_bytes": 0,
        "peak_transient_candidate_bytes": 0,
        "opaque_native_object_count": 0,
        "retained_bytes_accounting_complete": True,
        "quarantined_retained_bytes": 0,
        "pre_candidate_headroom_phase": {
            "calls": 0,
            "slow": 0,
            "total_duration_micros": 0,
            "max_duration_micros": 0,
        },
        "candidate_warm_phase": {
            "calls": 0,
            "slow": 0,
            "total_duration_micros": 0,
            "max_duration_micros": 0,
        },
        "pre_native_reservation_phase": {
            "calls": 0,
            "slow": 0,
            "total_duration_micros": 0,
            "max_duration_micros": 0,
        },
        "native_capture_phase": {
            "calls": 0,
            "slow": 0,
            "total_duration_micros": 0,
            "max_duration_micros": 0,
        },
        "rejected_candidate_cleanup_phase": {
            "calls": 0,
            "slow": 0,
            "total_duration_micros": 0,
            "max_duration_micros": 0,
        },
        "fallbacks": {
            "total": 0,
            "cold_cache_host_round_trip": 0,
            "persistent_host_round_trip": 0,
            "shape_dependent_attention": 0,
            "graph_cache_capacity": 0,
            "graph_cache_byte_budget": 0,
            "graph_accounting_incomplete": 0,
            "moderate_memory_pressure": 0,
            "tight_memory_pressure": 0,
            "critical_memory_pressure": 0,
            "memory_reservation_denied": 0,
            "memory_governor_selector_mismatch": 0,
            "capture_failure": 0,
            "replay_failure": 0,
            "slow": 0,
            "total_duration_micros": 0,
            "max_duration_micros": 0,
        },
    }
    accelerator_runtime = {
        "schema_id": "kiln.accelerator-runtime-policy.v2",
        "version": 2,
        "serving_profile": serving_profile,
        "serving_profile_source": "config_file",
        "rocm_synchronization_mode": {
            "configured": "legacy_host_barriers",
            "effective": "legacy_host_barriers",
            "source": "config_file",
        },
        "rocm_graph_mode": {
            "configured": "profile" if rocm_graphs_requested else "disabled",
            "effective": "lazy_capture_replay" if rocm_graphs else "disabled",
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
    return {
        "backend": "model",
        "backend_runtime": {"external_yield_sync": []},
        "serving_profile": {
            "profile": serving_profile,
            "source": "config_file",
            "immutable_after_startup": True,
            "request_overrides_allowed": False,
            "effective_policy_source": "serving_profile",
            "effective_policy": serve.PROFILE_POLICIES[serving_profile],
        },
        "gpu_memory": {
            "total_vram_bytes": 64 * 1024 * 1024 * 1024,
            "live": {"used_gb": 1.0, "source": "linux-drm-sysfs"},
        },
        "http": http_fixture(),
        "decode_runtime": {
            "accelerator_runtime": accelerator_runtime,
            "rocm_graphs": graph,
            "kv_autoscaler": {
                "requested": kv_autoscale_requested,
                "requested_source": "config_file",
                "force_blocks": None,
                "force_blocks_source": "config_file",
                "enabled": kv_autoscale,
                "state": (
                    "unavailable"
                    if serving_profile == "stable"
                    else "enabled" if kv_autoscale else "disabled"
                ),
                "reason": (
                    "serving_profile_stable"
                    if serving_profile == "stable"
                    else "active" if kv_autoscale else "configuration"
                ),
            },
            "memory_governor": {
                "reclaim_mode": memory_reclaim_mode,
                "requested_reclaim_mode": memory_reclaim_requested_mode,
                "automatic_monitor_enabled": memory_reclaim_mode == "automatic",
                "source": "config_file",
                "disabled_by_serving_profile": serving_profile == "stable",
            },
            "batching_engine": {
                "stream_stall_grace_ms": serve.STREAM_STALL_GRACE_MS,
                "stream_stall_grace_source": "config_file",
                "max_prefill_tokens_per_cycle": serve.MAX_PREFILL_TOKENS_PER_CYCLE,
                "max_prefill_tokens_per_cycle_source": "config_file",
                "max_prefill_layers_per_cycle": serve.MAX_PREFILL_LAYERS_PER_CYCLE,
                "max_prefill_layers_per_cycle_source": "config_file",
                "active_decode": 0,
                "active_prefill": 0,
                "active_staged_requests": 0,
                "queue_depth": 0,
                "max_decode_batch": serve.MAX_DECODE_BATCH,
                "max_prefill_staging_slots": serve.MAX_PREFILL_STAGING_SLOTS,
                "max_active_requests": serve.MAX_ACTIVE_REQUESTS,
                "max_prefill_staging_priority_burst": (
                    serve.MAX_PREFILL_STAGING_PRIORITY_BURST
                ),
                "max_observed_active_requests": serve.MAX_DECODE_BATCH,
                "max_observed_batch_size": 8,
                "total_errors": 0,
                "total_decode_forwards": 0,
                "total_batched_decode_forwards": 0,
                "total_decode_rows": 0,
                "total_decode_forward_ms": 0.0,
                "max_decode_forward_ms": 0.0,
                "slow_decode_forward_count": 0,
                "total_prefill_forwards": 0,
                "total_prefill_layers": 0,
                "total_prefill_layer_yields": 0,
                "total_short_prefill_priority_forwards": 0,
                "total_prefill_staging_admissions": 0,
                "total_prefill_staging_priority_forwards": 0,
                "total_prefill_forward_ms": 0.0,
                "max_prefill_forward_ms": 0.0,
                "slow_prefill_forward_count": 0,
                "total_admission_calls": 0,
                "total_admission_ms": 0.0,
                "max_admission_ms": 0.0,
                "slow_admission_count": 0,
                "response_backpressure_events": 0,
                "response_backpressure_wait_ms": 0,
                "response_stall_evictions": 0,
                "response_channel_closed": 0,
            },
        },
        "scheduler": {"blocks_total": 100, "blocks_used": 0},
    }


def debug_fixture(
    *,
    kv_autoscale: bool,
    rocm_graphs: bool,
    serving_profile: str = "experimental",
    rocm_graphs_enabled: bool | None = None,
    memory_reclaim_requested_mode: str = "off",
) -> dict:
    rocm_graphs_enabled = (
        rocm_graphs if rocm_graphs_enabled is None else rocm_graphs_enabled
    )
    health_graph = health_fixture(
        kv_autoscale=kv_autoscale,
        rocm_graphs=rocm_graphs_enabled,
        serving_profile=serving_profile,
        rocm_graphs_requested=rocm_graphs,
        memory_reclaim_requested_mode=memory_reclaim_requested_mode,
    )["decode_runtime"]["rocm_graphs"]
    health_only_fields = {
        "state",
        "unavailable_reason",
        "phase_telemetry_available",
        "phase_telemetry_unavailable_reason",
        "current_phase",
        "current_phase_elapsed_micros",
    }
    telemetry_fields = {
        "current_phase",
        "current_phase_elapsed_micros",
        "pre_candidate_headroom_phase",
        "candidate_warm_phase",
        "pre_native_reservation_phase",
        "native_capture_phase",
        "rejected_candidate_cleanup_phase",
        "last_transient_candidate_bytes",
        "peak_transient_candidate_bytes",
    }

    return {
        "accelerator_runtime": {
            "schema_id": "kiln.accelerator-runtime-policy.v2",
            "version": 2,
            "serving_profile": serving_profile,
            "serving_profile_source": "config_file",
            "rocm_synchronization_mode": {
                "configured": "legacy_host_barriers",
                "effective": "legacy_host_barriers",
                "source": "config_file",
            },
            "rocm_graph_mode": {
                "configured": "profile" if rocm_graphs else "disabled",
                "effective": (
                    "lazy_capture_replay" if rocm_graphs_enabled else "disabled"
                ),
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
        },
        "rocm_graphs": {
            field: value
            for field, value in health_graph.items()
            if field not in health_only_fields
        },
        "rocm_graphs_unavailable_reason": None,
        "rocm_graph_telemetry": {
            field: value for field, value in health_graph.items() if field in telemetry_fields
        },
        "rocm_graph_telemetry_unavailable_reason": None,
        "kv_autoscaler": health_fixture(
            kv_autoscale=kv_autoscale and serving_profile != "stable",
            rocm_graphs=rocm_graphs_enabled,
            serving_profile=serving_profile,
            kv_autoscale_requested=kv_autoscale,
            rocm_graphs_requested=rocm_graphs,
            memory_reclaim_requested_mode=memory_reclaim_requested_mode,
        )["decode_runtime"]["kv_autoscaler"],
        "http": http_fixture(),
        "batching_engine": {
            "backend": "model",
            "enabled": True,
            "snapshot": {
                "stream_stall_grace_ms": serve.STREAM_STALL_GRACE_MS,
                "stream_stall_grace_source": "config_file",
                "max_prefill_tokens_per_cycle": serve.MAX_PREFILL_TOKENS_PER_CYCLE,
                "max_prefill_tokens_per_cycle_source": "config_file",
                "max_prefill_layers_per_cycle": serve.MAX_PREFILL_LAYERS_PER_CYCLE,
                "max_prefill_layers_per_cycle_source": "config_file",
                "max_decode_batch": serve.MAX_DECODE_BATCH,
                "max_prefill_staging_slots": serve.MAX_PREFILL_STAGING_SLOTS,
                "max_active_requests": serve.MAX_ACTIVE_REQUESTS,
                "max_prefill_staging_priority_burst": (
                    serve.MAX_PREFILL_STAGING_PRIORITY_BURST
                ),
            },
        },
        "env_flags": {
            "KILN_MEMORY_RECLAIM_MODE": {
                "present": False,
                "value": None,
            },
            "KILN_HTTP_SEND_BUFFER_BYTES": {
                "present": False,
                "value": None,
            },
            "KILN_STREAM_STALL_GRACE_MS": {
                "present": False,
                "value": None,
            },
            "KILN_MAX_PREFILL_TOKENS_PER_CYCLE": {
                "present": False,
                "value": None,
            },
            "KILN_MAX_PREFILL_LAYERS_PER_CYCLE": {
                "present": False,
                "value": None,
            },
        }
    }


class ServeMixedLoadTests(unittest.TestCase):
    def test_serving_run_directories_are_private_and_namespace_collision_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "serving"
            first = serve.create_serving_run_dir("lifecycle", parent=parent)
            second = serve.create_serving_run_dir("lifecycle", parent=parent)

            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, parent)
            self.assertEqual(second.parent, parent)
            self.assertTrue(first.name.startswith("lifecycle-"))
            self.assertTrue(second.name.startswith("lifecycle-"))
            self.assertEqual(first.stat().st_mode & 0o777, 0o700)
            self.assertEqual(second.stat().st_mode & 0o777, 0o700)

        with self.assertRaisesRegex(serve.QualificationError, "prefix"):
            serve.create_serving_run_dir("../escape")

    def test_wait_ready_refreshes_health_after_prewarm_log(self) -> None:
        def ready_health(generation: int) -> dict[str, object]:
            return {
                "status": "ok",
                "checks": [
                    {"name": "model_loaded", "pass": True},
                    {"name": "inference_prewarm_complete", "pass": True},
                ],
                "generation": generation,
            }

        process = mock.Mock()
        process.poll.return_value = None
        server_log = mock.Mock()
        server_log.prewarm_complete = serve.threading.Event()
        server_log.prewarm_complete.set()

        with mock.patch.object(
            serve,
            "json_request",
            side_effect=[ready_health(1), ready_health(2)],
        ) as request:
            health = serve.wait_ready(
                1234,
                process,
                server_log,
                serve.time.monotonic() + 5.0,
            )

        self.assertEqual(health["generation"], 2)
        self.assertEqual(request.call_count, 2)

    def test_stream_reader_waits_for_readiness_before_touching_http_buffer(self) -> None:
        sock = mock.Mock()
        connection = mock.Mock(sock=sock)
        response = mock.Mock()
        response.fp.peek.return_value = b""
        response.read1.return_value = b"data: {}\n\n"

        with mock.patch.object(
            serve.select,
            "select",
            side_effect=[([], [], []), ([sock], [], [])],
        ) as select_call:
            chunk = serve.read_stream_chunk(
                connection,
                response,
                deadline=serve.time.monotonic() + 10.0,
                abort_event=None,
                name="reader-test",
            )

        self.assertEqual(chunk, b"data: {}\n\n")
        self.assertEqual(select_call.call_count, 2)
        response.read1.assert_called_once_with(4096)

    def test_stream_reader_drains_buffered_body_before_polling_socket(self) -> None:
        sock = mock.Mock()
        connection = mock.Mock(sock=sock)
        response = mock.Mock()
        response.fp.peek.return_value = b"data"
        response.read1.return_value = b"data: {}\n\n"

        with mock.patch.object(serve.select, "select") as select_call:
            chunk = serve.read_stream_chunk(
                connection,
                response,
                deadline=serve.time.monotonic() + 10.0,
                abort_event=None,
                name="reader-test",
            )

        self.assertEqual(chunk, b"data: {}\n\n")
        select_call.assert_not_called()
        sock.setblocking.assert_called_once_with(False)
        response.fp.peek.assert_called_once_with(1)
        response.read1.assert_called_once_with(4096)

    def test_stream_reader_drains_real_httpresponse_buffer_on_keep_alive(self) -> None:
        body = b"data: [DONE]\n\n"
        encoded = f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"
        listener = serve.socket.socket(serve.socket.AF_INET, serve.socket.SOCK_STREAM)
        listener.setsockopt(serve.socket.SOL_SOCKET, serve.socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(2.0)
        port = listener.getsockname()[1]
        release = serve.threading.Event()

        def serve_once() -> None:
            accepted, _ = listener.accept()
            accepted.settimeout(2.0)
            request = b""
            while b"\r\n\r\n" not in request:
                request += accepted.recv(4096)
            accepted.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"Connection: keep-alive\r\n\r\n"
                + encoded
            )
            release.wait(2.0)
            accepted.close()

        worker = serve.threading.Thread(target=serve_once, daemon=True)
        worker.start()
        connection = serve.http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
        try:
            connection.request("GET", "/stream")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(response.fp.peek(1))

            with mock.patch.object(
                serve.select,
                "select",
                side_effect=AssertionError("buffered body must not poll the socket"),
            ):
                chunk = serve.read_stream_chunk(
                    connection,
                    response,
                    deadline=serve.time.monotonic() + 2.0,
                    abort_event=None,
                    name="real-buffer-test",
                )
            self.assertEqual(chunk, body)
        finally:
            release.set()
            connection.close()
            listener.close()
            worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())

    def test_stream_reader_retries_interrupted_readiness_poll(self) -> None:
        sock = mock.Mock()
        connection = mock.Mock(sock=sock)
        response = mock.Mock()
        response.fp.peek.return_value = b""
        response.read1.return_value = b""

        with mock.patch.object(
            serve.select,
            "select",
            side_effect=[InterruptedError(), ([sock], [], [])],
        ):
            self.assertEqual(
                serve.read_stream_chunk(
                    connection,
                    response,
                    deadline=serve.time.monotonic() + 10.0,
                    abort_event=None,
                    name="reader-test",
                ),
                b"",
            )

    def test_stream_reader_honors_cleanup_without_buffered_read_timeout(self) -> None:
        abort = serve.threading.Event()
        abort.set()
        connection = mock.Mock(sock=object())
        response = mock.Mock()

        with mock.patch.object(serve.select, "select") as select_call:
            with self.assertRaisesRegex(
                serve.QualificationError, "aborted by qualification cleanup"
            ):
                serve.read_stream_chunk(
                    connection,
                    response,
                    deadline=serve.time.monotonic() + 10.0,
                    abort_event=abort,
                    name="reader-test",
                )

        select_call.assert_not_called()
        response.read1.assert_not_called()

    def test_stream_reader_enforces_request_deadline_before_read(self) -> None:
        connection = mock.Mock(sock=object())
        response = mock.Mock()

        with self.assertRaisesRegex(TimeoutError, "exceeded its request or overall deadline"):
            serve.read_stream_chunk(
                connection,
                response,
                deadline=serve.time.monotonic() - 1.0,
                abort_event=None,
                name="reader-test",
            )

        response.read1.assert_not_called()

    def test_run_stream_uses_blocking_http_buffer_after_readiness_poll(self) -> None:
        sock = mock.Mock()
        connection = mock.Mock(sock=sock)
        response = mock.Mock(status=200)
        response.getheader.side_effect = lambda name, default=None: {
            "Content-Type": "text/event-stream",
            "X-Kiln-Loaded-Adapter": "fixture-adapter",
            "X-Kiln-Loaded-Adapter-Revision": "a" * 64,
        }.get(name, default)
        response.fp.peek.return_value = b"data"
        response.read1.return_value = (
            b'data: {"choices":[{"delta":{"content":"token"}}]}\n\n'
        )
        connection.getresponse.return_value = response

        with (
            mock.patch.object(
                serve.http.client, "HTTPConnection", return_value=connection
            ),
            mock.patch.object(
                serve.select, "select", return_value=([sock], [], [])
            ),
        ):
            result = serve.run_stream(
                12345,
                name="reader-test",
                marker="marker",
                prompt_words=1,
                max_tokens=2,
                seed=7,
                cancel_after=1,
            )

        self.assertTrue(result.cancelled)
        self.assertEqual(len(result.semantic_times), 1)
        self.assertEqual(len(result.semantic_deltas), 1)
        self.assertEqual(result.loaded_adapter, "fixture-adapter")
        self.assertEqual(result.loaded_adapter_revision, "a" * 64)
        sock.setblocking.assert_called_once_with(False)
        sock.settimeout.assert_called_once()
        response.read1.assert_called_once_with(4096)
        connection.close.assert_called_once_with()

    def test_run_stream_preserves_structured_generation_error(self) -> None:
        sock = mock.Mock()
        connection = mock.Mock(sock=sock)
        response = mock.Mock(status=200)
        response.getheader.side_effect = lambda name, default=None: {
            "Content-Type": "text/event-stream",
        }.get(name, default)
        response.fp.peek.return_value = b"data"
        response.read1.return_value = (
            b'data: {"error":{"message":"device copy failed","type":"server_error",'
            b'"code":"generation_error"}}\n\ndata: [DONE]\n\n'
        )
        connection.getresponse.return_value = response

        with mock.patch.object(
            serve.http.client, "HTTPConnection", return_value=connection
        ):
            result = serve.run_stream(
                12345,
                name="error-test",
                marker="marker",
                prompt_words=1,
                max_tokens=2,
                seed=7,
            )

        self.assertEqual(
            result.error,
            "QualificationError: error-test stream generation_error: device copy failed",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.semantic_deltas, [])
        connection.close.assert_called_once_with()

    def test_checked_in_workload_exactly_matches_driver_contract(self) -> None:
        manifest = json.loads(
            (ROOT / "qualification/workloads/serving-mixed-rocm-v1.json").read_text()
        )
        variants = {variant["id"]: variant for variant in manifest["variants"]}
        self.assertEqual(list(variants), sorted(serve.VARIANT_CONFIGS))
        self.assertEqual(set(variants), set(serve.VARIANT_CONFIGS))

        expected_command = [
            "python3",
            "scripts/qualification/serve_mixed_load.py",
            "--model-path",
            "${model_path}",
            "--seed",
            "${seed}",
        ]
        expected_build = {
            "binary": "kiln",
            "cargo_jobs": 1,
            "cargo_execution_mode": "transient-service",
            "cargo_environment_policy": "closed-source-build-v1",
            "cargo_memory_scope": "systemd_user_transient_service_memory_max_no_swap",
            "cargo_min_available_gib": 15,
            "cargo_private_network": True,
            "cargo_service_runtime_max_seconds": 840,
            "cargo_wrapper": "scripts/cargo-bounded.sh",
            "features": "rocm",
            "locked": True,
            "no_default_features": True,
            "offline": True,
            "package": "kiln-server",
            "profile": "release",
            "timeout_seconds": 900,
            "rocm_archs": "gfx1151",
            "rocm_path": "/opt/rocm",
        }
        expected_schedule = {
            "cancellation_after_semantic_deltas": serve.CANCELLATION_AFTER_DELTAS,
            "long_prefill_max_tokens": serve.LONG_PREFILL_MAX_TOKENS,
            "long_prefill_words": serve.LONG_PREFILL_WORDS,
            "max_warmup_requests": serve.MAX_WARMUP_REQUESTS,
            "measured_expected_completion_tokens": serve.MEASURED_EXPECTED_COMPLETION_TOKENS,
            "measured_finish_reason": "length",
            "measured_ignore_eos": True,
            "memory_poll_interval_ms": int(serve.MEMORY_POLL_INTERVAL_SECONDS * 1000),
            "normal_max_tokens": serve.NORMAL_MAX_TOKENS,
            "normal_requests": serve.NORMAL_REQUESTS,
            "outlier_absolute_ms": int(serve.OUTLIER_ABSOLUTE_MS),
            "outlier_history_size": serve.OUTLIER_HISTORY_SIZE,
            "outlier_multiplier": int(serve.OUTLIER_MULTIPLIER),
            "overall_timeout_seconds": int(serve.OVERALL_TIMEOUT_SECONDS),
            "pressure_peer_dispatch": "after_slow_headers_before_pressure_wait",
            "pressure_peer_max_tokens": serve.PRESSURE_PEER_MAX_TOKENS,
            "pressure_peer_prompt_words": serve.PRESSURE_PEER_PROMPT_WORDS,
            "pressure_peer_seed_offset": serve.PRESSURE_PEER_SEED_OFFSET,
            "prompt_identity": serve.PROMPT_IDENTITY,
            "prompt_marker_format": serve.PROMPT_MARKER_FORMAT,
            "request_timeout_seconds": int(serve.REQUEST_TIMEOUT_SECONDS),
            "slow_socket_buffer_bytes": serve.SLOW_SOCKET_BUFFER_BYTES,
            "slow_max_tokens": serve.SLOW_MAX_TOKENS,
            "startup_timeout_seconds": int(serve.STARTUP_TIMEOUT_SECONDS),
            "warmup_max_tokens": serve.WARMUP_MAX_TOKENS,
        }
        expected_metrics = sorted(serve.METRIC_DEFINITIONS)
        for variant_id, driver_config in serve.VARIANT_CONFIGS.items():
            with self.subTest(variant=variant_id):
                declared = variants[variant_id]
                self.assertEqual(declared["effective_config"], driver_config)
                self.assertEqual(declared["effective_config"]["build"], expected_build)
                self.assertEqual(declared["effective_config"]["workload"], expected_schedule)
                self.assertEqual(len(declared["cases"]), 1)
                case = declared["cases"][0]
                self.assertEqual(case["id"], serve.CASE_ID)
                self.assertEqual(case["command"], expected_command)
                self.assertEqual(
                    case["result_protocol"],
                    {
                        "format": "qualification-case-result-v1",
                        "producer": "command",
                        "path_environment_variable": serve.RESULT_ENV,
                        "declared_metrics": expected_metrics,
                    },
                )

        self.assertEqual(
            serve.source_bound_build_command(),
            [
                str(ROOT / "scripts/cargo-bounded.sh"),
                "build",
                "--quiet",
                "--release",
                "--locked",
                "--offline",
                "-p",
                "kiln-server",
                "--bin",
                "kiln",
                "--no-default-features",
                "--features",
                "rocm",
            ],
        )
        self.assertEqual(
            serve.source_bound_build_command(serve.VULKAN_BUILD_SPEC),
            [
                str(ROOT / "scripts/cargo-bounded.sh"),
                "build",
                "--quiet",
                "--release",
                "--locked",
                "--offline",
                "-p",
                "kiln-server",
                "--bin",
                "kiln",
                "--no-default-features",
                "--features",
                "vulkan",
            ],
        )

        parsed = serve.parse_args(["--model-path", "/models/test", "--seed", "7"])
        self.assertEqual(parsed.model_path, Path("/models/test"))
        self.assertEqual(parsed.seed, 7)

        pairs = manifest["comparison_policy"]["variant_pairs"]
        self.assertEqual(
            {
                (pair["baseline_variant_id"], pair["candidate_variant_id"])
                for pair in pairs
            },
            {
                ("default", variant_id)
                for variant_id in serve.VARIANT_CONFIGS
                if variant_id not in {"default", "stable"}
            },
        )

    def test_cargo_resolution_uses_rustup_home_when_path_omits_cargo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cargo = Path(temp_dir) / ".cargo" / "bin" / "cargo"
            cargo.parent.mkdir(parents=True)
            cargo.write_text("#!/bin/sh\n")
            cargo.chmod(0o755)

            resolved = serve.resolve_cargo_executable(
                {"HOME": temp_dir, "PATH": "/usr/bin"}
            )

        self.assertEqual(resolved, str(cargo))

    def test_cargo_resolution_rejects_invalid_explicit_override(self) -> None:
        with self.assertRaisesRegex(serve.QualificationError, "CARGO=.*executable"):
            serve.resolve_cargo_executable(
                {"CARGO": "/missing/cargo", "HOME": "/missing", "PATH": ""}
            )

    def test_source_bound_build_environment_forces_bounded_offline_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cargo = Path(temp_dir) / "cargo"
            cargo.write_text("#!/bin/sh\n")
            cargo.chmod(0o755)
            environment = serve.source_bound_build_environment(
                {
                    "CARGO": str(cargo),
                    "HOME": temp_dir,
                    "OPENAI_API_KEY": "must-not-enter-build",
                    "PATH": "/usr/bin",
                    serve.RESULT_ENV: "/tmp/result.json",
                    serve.VARIANT_ENV: "default",
                }
            )
        self.assertEqual(environment["CARGO"], str(cargo))
        self.assertEqual(environment["CARGO_NET_OFFLINE"], "true")
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertEqual(
            environment["KILN_CARGO_ENVIRONMENT_POLICY"], "closed-source-build-v1"
        )
        self.assertEqual(environment["KILN_CARGO_EXECUTION_MODE"], "transient-service")
        self.assertEqual(environment["KILN_CARGO_JOBS"], "1")
        self.assertEqual(environment["KILN_CARGO_MIN_AVAILABLE_GIB"], "15")
        self.assertEqual(environment["KILN_CARGO_PRIVATE_NETWORK"], "1")
        self.assertEqual(environment["KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS"], "840")
        self.assertEqual(environment["KILN_ROCM_ARCHS"], serve.BUILD_ROCM_ARCHS)

    def test_vulkan_source_build_is_bounded_without_rocm_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cargo = Path(temp_dir) / "cargo"
            cargo.write_text("#!/bin/sh\n")
            cargo.chmod(0o755)
            environment = serve.source_bound_build_environment(
                {
                    "CARGO": str(cargo),
                    "HOME": temp_dir,
                    "PATH": "/usr/bin",
                    "ROCM_PATH": "/ambient/rocm",
                    serve.RESULT_ENV: "/tmp/result.json",
                    serve.VARIANT_ENV: "vulkan-serving-baseline",
                },
                serve.VULKAN_BUILD_SPEC,
            )
        self.assertEqual(environment["CARGO"], str(cargo))
        self.assertEqual(environment["CARGO_NET_OFFLINE"], "true")
        self.assertEqual(environment["KILN_CARGO_JOBS"], "1")
        self.assertEqual(environment["KILN_CARGO_MIN_AVAILABLE_GIB"], "15")
        self.assertEqual(environment["KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS"], "840")
        self.assertNotIn("ROCM_PATH", environment)
        self.assertNotIn("KILN_ROCM_ARCHS", environment)
        self.assertEqual(
            serve.VULKAN_BUILD_SPEC.effective_config(),
            {
                "binary": "kiln",
                "cargo_jobs": 1,
                "cargo_execution_mode": "transient-service",
                "cargo_environment_policy": "closed-source-build-v1",
                "cargo_memory_scope": "systemd_user_transient_service_memory_max_no_swap",
                "cargo_min_available_gib": 15,
                "cargo_private_network": True,
                "cargo_service_runtime_max_seconds": 840,
                "cargo_wrapper": "scripts/cargo-bounded.sh",
                "features": "vulkan",
                "locked": True,
                "no_default_features": True,
                "offline": True,
                "package": "kiln-server",
                "profile": "release",
                "timeout_seconds": 900,
            },
        )

    def test_source_build_timeout_reserves_service_teardown_window(self) -> None:
        spec = serve.SourceBuildSpec(
            backend="test",
            features="test",
            cargo_service_runtime_max_seconds=300,
            timeout_seconds=359,
        )
        with self.assertRaisesRegex(serve.QualificationError, "at least 60 seconds"):
            serve.build_binary(1.0e18, spec)

    def test_vulkan_server_environment_cannot_inherit_rocm_toolchain(self) -> None:
        with mock.patch.dict(
            serve.os.environ,
            {
                "PATH": "/usr/bin",
                "ROCM_PATH": "/ambient/rocm",
                "HIP_PATH": "/ambient/hip",
                serve.RESULT_ENV: "/tmp/result.json",
                serve.VARIANT_ENV: "default",
            },
            clear=True,
        ):
            vulkan = serve.server_environment("default", serve.VULKAN_BUILD_SPEC)
            rocm = serve.server_environment("default")
        self.assertNotIn("ROCM_PATH", vulkan)
        self.assertNotIn("HIP_PATH", vulkan)
        self.assertEqual(rocm["ROCM_PATH"], serve.BUILD_ROCM_PATH)
        self.assertNotIn("HIP_PATH", rocm)

    def test_source_bound_build_environment_rejects_wrapper_recursion(self) -> None:
        with self.assertRaisesRegex(serve.QualificationError, "must name the cargo"):
            serve.source_bound_build_environment(
                {
                    "CARGO": str(ROOT / "scripts/cargo-bounded.sh"),
                    "HOME": str(ROOT),
                    "PATH": "/usr/bin",
                    serve.RESULT_ENV: "/tmp/result.json",
                    serve.VARIANT_ENV: "default",
                }
            )

    def test_sse_parser_handles_fragmentation_crlf_comments_and_multiline_data(self) -> None:
        parser = serve.SSEParser()
        chunks = [
            b": keepalive\r",
            b"\ndata: {\"a\":",
            b"1}\r\n\r\ndata: first\n",
            b"data: second\n\n",
        ]
        events = []
        for chunk in chunks:
            events.extend(parser.feed(chunk))
        self.assertEqual(events, ['{"a":1}', "first\nsecond"])

    def test_semantic_delta_excludes_role_finish_and_usage(self) -> None:
        self.assertFalse(serve.semantic_delta({"choices": [{"delta": {"role": "assistant"}}]}))
        self.assertFalse(
            serve.semantic_delta(
                {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {}}
            )
        )
        self.assertTrue(
            serve.semantic_delta({"choices": [{"delta": {"content": "answer"}}]})
        )
        self.assertTrue(
            serve.semantic_delta(
                {"choices": [{"delta": {"reasoning_content": "thought"}}]}
            )
        )
        self.assertTrue(
            serve.semantic_delta({"choices": [{"delta": {"tool_calls": [{}]}}]})
        )

    def test_token_timing_parser_is_strict_and_ordered(self) -> None:
        timing = {
            "object": "kiln.token_timing",
            "source": "batching_engine",
            "token_index": 1,
            "ready_ms": 12.0,
            "producer_delivered_ms": 14.0,
            "handler_received_ms": 17.0,
            "body_enqueued_ms": 20.0,
            "response_delivery_ms": 2.0,
            "handler_queue_ms": 3.0,
            "queue_delay_ms": 5.0,
            "client_delivery_ms": 3.0,
            "blocking_phase": None,
            "blocking_phase_ms": None,
        }
        self.assertEqual(serve.parse_token_timing(timing, 1), (12.0, 5.0))
        self.assertIsNone(serve.parse_token_timing({"choices": []}, 1))

        timing["token_index"] = 2
        with self.assertRaises(serve.QualificationError):
            serve.parse_token_timing(timing, 1)
        timing["token_index"] = 1
        timing["queue_delay_ms"] = 4.0
        with self.assertRaises(serve.QualificationError):
            serve.parse_token_timing(timing, 1)
        timing["queue_delay_ms"] = 5.0

        for invalid_index in (True, 1.0):
            timing["token_index"] = invalid_index
            with self.subTest(token_index=invalid_index):
                with self.assertRaises(serve.QualificationError):
                    serve.parse_token_timing(timing, 1)
        timing.update({
            "token_index": 2,
            "ready_ms": 20.0,
            "producer_delivered_ms": 22.0,
            "handler_received_ms": 25.0,
            "body_enqueued_ms": 28.0,
            "blocking_phase": "actor_decode",
            "blocking_phase_ms": 8.0,
        })
        self.assertEqual(
            serve.parse_token_timing(timing, 2, previous_ready_ms=12.0),
            (20.0, 5.0),
        )
        with self.assertRaises(serve.QualificationError):
            serve.parse_token_timing(timing, 2, previous_ready_ms=21.0)
        timing["source"] = "unbounded"
        with self.assertRaises(serve.QualificationError):
            serve.parse_token_timing(timing, 2, previous_ready_ms=12.0)

    def test_token_timing_usage_contract_allows_only_consumed_eos_delta(self) -> None:
        self.assertTrue(serve.token_timing_matches_usage("length", 3, 3))
        self.assertFalse(serve.token_timing_matches_usage("length", 2, 3))

        self.assertTrue(serve.token_timing_matches_usage("stop", 3, 3))
        self.assertTrue(serve.token_timing_matches_usage("stop", 3, 4))
        self.assertFalse(serve.token_timing_matches_usage("stop", 2, 4))
        self.assertFalse(serve.token_timing_matches_usage("stop", 0, 1))
        self.assertFalse(serve.token_timing_matches_usage(None, 3, 3))

    def test_actor_performance_parser_is_strict_and_bounded_by_ttft(self) -> None:
        performance = {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "ttft_ms": 40.0,
            "prefill_ms": 10.0,
            "actor_queue_ms": 12.0,
            "actor_admission_ms": 3.0,
            "actor_prefill_wall_ms": 20.0,
            "decode_ms": 8.0,
            "total_latency_ms": 60.0,
            "decode_tokens_per_sec": 375.0,
            "adapter_used": "base",
            "thinking_mode": "non_reasoning",
            "finish_reason": "length",
            "latency": {
                "emitted_tokens": 3,
                "gap_samples": 2,
                "retained_gap_samples": 2,
                "gap_samples_truncated": False,
                "ttft_ms": 40.0,
                "itl_ms_p50": 8.0,
                "itl_ms_p99": 8.0,
                "itl_ms_p999": 8.0,
                "max_itl_ms": 8.0,
                "stall_threshold_ms": 250.0,
                "stall_count": 0,
                "unexplained_stall_count": 0,
                "stall_reasons": {
                    "actor_queue": 0,
                    "actor_admission": 0,
                    "actor_prefill": 0,
                    "actor_decode": 0,
                    "response_delivery": 0,
                    "handler_queue": 0,
                    "client_delivery": 0,
                    "sampling": 0,
                    "readback": 0,
                    "gpu_lock_wait": 0,
                    "graph_capture": 0,
                    "graph_replay": 0,
                    "synchronization": 0,
                    "resize": 0,
                    "trim": 0,
                    "adapter": 0,
                    "training": 0,
                    "unexplained": 0,
                },
                "phases": {
                    "actor_queue_ms": 12.0,
                    "actor_admission_ms": 3.0,
                    "tokenization_ms": 1.0,
                    "prefill_ms": 20.0,
                    "decode_ms": 8.0,
                    "sampling_ms": None,
                    "readback_ms": None,
                    "response_delivery_ms": 1.0,
                    "handler_queue_ms": 1.0,
                    "client_delivery_ms": 1.0,
                    "gpu_lock_wait_ms": None,
                    "graph_capture_ms": None,
                    "graph_replay_ms": None,
                    "synchronization_ms": None,
                    "resize_ms": None,
                    "trim_ms": None,
                    "adapter_ms": None,
                    "training_ms": None,
                    "unexplained_ms": 0.0,
                },
            },
        }
        value = {"metadata": {"performance": performance}}
        self.assertEqual(
            serve.parse_actor_performance(value),
            (12.0, 3.0, 20.0),
        )
        self.assertIsNone(serve.parse_actor_performance({"choices": []}))

        malformed = dict(performance)
        malformed["unexpected"] = 1
        with self.assertRaisesRegex(serve.QualificationError, "unexpected shape"):
            serve.parse_actor_performance({"metadata": {"performance": malformed}})

        impossible = dict(performance)
        impossible["actor_prefill_wall_ms"] = 30.0
        with self.assertRaisesRegex(serve.QualificationError, "exceed TTFT"):
            serve.parse_actor_performance({"metadata": {"performance": impossible}})

        impossible = dict(performance)
        impossible["prefill_ms"] = 24.0
        impossible["actor_admission_ms"] = 1.0
        impossible["actor_prefill_wall_ms"] = 20.0
        with self.assertRaisesRegex(serve.QualificationError, "prefill time exceeds"):
            serve.parse_actor_performance({"metadata": {"performance": impossible}})

    def test_percentile_uses_r7_linear_interpolation(self) -> None:
        self.assertEqual(serve.percentile_r7([], 0.99), 0.0)
        self.assertEqual(serve.percentile_r7([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(serve.percentile_r7([0, 100], 0.99), 99.0)
        with self.assertRaises(ValueError):
            serve.percentile_r7([1], 1.1)

    def test_prometheus_parser_selects_used_memory_only(self) -> None:
        payload = """
# TYPE kiln_gpu_memory_bytes gauge
kiln_gpu_memory_bytes{kind="total"} 128000000000
kiln_gpu_memory_bytes{kind="used"} 123456789
kiln_gpu_memory_bytes{kind="free"} 127876543211
"""
        self.assertEqual(serve.parse_prometheus_used_bytes(payload), 123456789)
        self.assertIsNone(serve.parse_prometheus_used_bytes("other_metric 1\n"))

    def test_server_event_classifier_is_causal_and_narrow(self) -> None:
        cases = {
            "background inference prewarm complete": "prewarm_complete",
            "Vulkan decode weight prewarm complete": "prewarm_complete",
            "KV autoscaler resized cache": "kv_resize",
            "KV cache physical resize completed": "kv_resize",
            "ROCm pool reclaim completed": "memory_reclaim",
            "memory governor automatic reclaim completed": "memory_reclaim",
            "ROCm HIP graph captured for decode (24 layers)": "graph_capture",
            "ROCm graph capture failed: bad launch": "graph_fallback",
            "slow_backend_external_yield_sync": "external_yield_sync",
            "hipErrorLaunchFailure": "device_fault",
            "ROCm graph replay failed: hipErrorIllegalAddress": "device_fault",
            "GPU memory access fault on agent": "device_fault",
            "an illegal memory access was encountered": "device_fault",
            "hipMemcpy failed: unknown (hipError 700)": "device_fault",
            "HSA_STATUS_ERROR_EXCEPTION": "device_fault",
            "device lost while synchronizing": "device_fault",
            "response_channel_backpressure": "client_backpressure_start",
            "response_channel_backpressure_timeout": "client_backpressure_timeout",
            "stream_request_bound": "stream_request_bound",
            "ROCm graph runner: warmup decode step": None,
            "response channel full backpressure": None,
            "long prefill started": None,
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(serve.classify_server_event(message), expected)
        for phase in ("admission", "prefill", "decode"):
            with self.subTest(phase=phase):
                self.assertEqual(
                    serve.classify_server_event(
                        "slow_batching_actor_phase",
                        {"event": "slow_batching_actor_phase", "phase": phase},
                    ),
                    f"actor_{phase}",
                )
        self.assertIsNone(
            serve.classify_server_event(
                "slow_batching_actor_phase",
                {"event": "slow_batching_actor_phase", "phase": "unknown"},
            )
        )
        self.assertEqual(
            serve.classify_server_event(
                "batched real generation failed",
                {
                    "event": "generation_error",
                    "error": "hipGraphLaunch failed: launch failure (hipError 719)",
                },
            ),
            "device_fault",
        )
        structured_cases = [
            (
                "KV cache physical resize completed",
                {
                    "event": "gpu_memory_operation",
                    "operation": "resize",
                    "reason": "automatic_memory_policy",
                },
                "kv_resize",
            ),
            (
                "ROCm pool reclaim completed",
                {
                    "event": "gpu_memory_operation",
                    "operation": "trim",
                    "reason": "memory_governor",
                },
                "memory_reclaim",
            ),
            (
                "ROCm graph host synchronization completed",
                {
                    "event": "gpu_memory_operation",
                    "operation": "synchronize",
                    "reason": "rocm_graph_capture_begin",
                },
                "graph_sync",
            ),
            (
                "ROCm graph eager fallback activated",
                {
                    "event": "rocm_graph_fallback",
                    "reason": "critical_memory_pressure",
                },
                "graph_fallback",
            ),
            (
                "KV pool allocation completed",
                {
                    "event": "gpu_memory_operation",
                    "operation": "allocation",
                    "reason": "kv_physical_resize",
                },
                None,
            ),
            (
                "unknown synchronization",
                {
                    "event": "gpu_memory_operation",
                    "operation": "synchronize",
                    "reason": "operator_supplied_text",
                },
                None,
            ),
        ]
        for message, fields, expected in structured_cases:
            with self.subTest(message=message, fields=fields):
                self.assertEqual(
                    serve.classify_server_event(message, fields), expected
                )

    def test_structured_log_fields_bind_pressure_to_the_slow_request(self) -> None:
        line = json.dumps(
            {
                "fields": {
                    "message": "stream_request_bound",
                    "event": "stream_request_bound",
                    "request_id": "slow-id",
                    "client": "qualification-slow-marker",
                }
            }
        )
        message, fields = serve.parse_server_log_line(line)
        self.assertEqual(message, "stream_request_bound")
        self.assertEqual(fields["request_id"], "slow-id")
        self.assertEqual(
            serve.classify_server_event(message, fields), "stream_request_bound"
        )

        events = [
            serve.ObservedEvent(1.0, "stream_request_bound", message, fields),
            serve.ObservedEvent(
                1.5,
                "client_backpressure_start",
                "response_channel_backpressure",
                {"request_id": "unrelated"},
            ),
            serve.ObservedEvent(
                2.0,
                "client_backpressure_start",
                "response_channel_backpressure",
                {"request_id": "slow-id"},
            ),
            serve.ObservedEvent(
                3.0,
                "client_backpressure_timeout",
                "response_channel_backpressure_timeout",
                {"request_id": "unrelated"},
            ),
        ]
        self.assertIsNone(
            serve.attributed_delivery_pressure_window(
                events, "qualification-slow-marker"
            )
        )
        events.append(
            serve.ObservedEvent(
                4.0,
                "client_backpressure_timeout",
                "response_channel_backpressure_timeout",
                {"request_id": "slow-id"},
            )
        )
        pressure = serve.attributed_delivery_pressure_window(
            events, "qualification-slow-marker"
        )
        self.assertEqual(
            pressure,
            serve.DeliveryPressureWindow(
                request_id="slow-id",
                client="qualification-slow-marker",
                started=2.0,
                timed_out=4.0,
            ),
        )

    def test_slow_socket_applies_receive_controls_before_connect(self) -> None:
        class FakeSocket:
            def __init__(self) -> None:
                self.events: list[tuple] = []

            def settimeout(self, value: float) -> None:
                self.events.append(("timeout", value))

            def setsockopt(self, level: int, option: int, value: int) -> None:
                self.events.append(("setsockopt", level, option, value))

            def connect(self, address: tuple[str, int]) -> None:
                self.events.append(("connect", address))

            def close(self) -> None:
                self.events.append(("close",))

        fake = FakeSocket()
        connected = serve.connect_slow_consumer_socket(
            8420, socket_factory=lambda *_: fake
        )
        self.assertIs(connected, fake)
        self.assertEqual(
            [event[0] for event in fake.events],
            ["timeout", "setsockopt", "setsockopt", "connect"],
        )
        self.assertEqual(
            fake.events[1],
            (
                "setsockopt",
                serve.socket.SOL_SOCKET,
                serve.socket.SO_RCVBUF,
                serve.SLOW_SOCKET_BUFFER_BYTES,
            ),
        )
        self.assertEqual(fake.events[2][2], serve.socket.TCP_WINDOW_CLAMP)

        unsupported = FakeSocket()
        with mock.patch.object(serve.socket, "TCP_WINDOW_CLAMP", None):
            with self.assertRaises(serve.QualificationError):
                serve.connect_slow_consumer_socket(
                    8420, socket_factory=lambda *_: unsupported
                )
        self.assertEqual(unsupported.events[-1], ("close",))

    def test_dedicated_pressure_peer_must_overlap_attributed_window(self) -> None:
        pressure = serve.DeliveryPressureWindow(
            request_id="slow-id",
            client="qualification-slow-marker",
            started=2.0,
            timed_out=4.0,
        )

        def result(
            name: str,
            started: float,
            finished: float,
            token_ready_times: list[float],
        ) -> serve.StreamResult:
            return serve.StreamResult(
                name=name,
                marker=name,
                started=started,
                finished=finished,
                semantic_times=token_ready_times,
                token_ready_times=token_ready_times,
                token_queue_delays_ms=[0.0] * len(token_ready_times),
                prompt_tokens=1,
                completion_tokens=len(token_ready_times),
                usage_records=1,
                finish_reason="length",
                done=True,
                cancelled=False,
                error=None,
                actor_queue_ms=100.0,
                actor_admission_ms=10.0,
                actor_prefill_wall_ms=200.0,
            )

        self.assertFalse(
            serve.healthy_peer_overlaps_pressure(
                result("before", 0.0, 1.0, [0.1]), pressure
            )
        )
        self.assertTrue(
            serve.healthy_peer_overlaps_pressure(
                result("pressure-peer", 1.5, 4.5, [1.9, 3.0, 4.1]), pressure
            )
        )
        timing = serve.pressure_peer_timing_values(
            result("pressure-peer", 1.5, 4.5, [1.9, 3.0, 4.1]), pressure
        )
        self.assertAlmostEqual(
            timing["pressure_peer_first_ready_after_dispatch_ms"], 400.0
        )
        self.assertEqual(timing["pressure_peer_actor_queue_ms"], 100.0)
        self.assertEqual(timing["pressure_peer_actor_admission_ms"], 10.0)
        self.assertEqual(timing["pressure_peer_actor_prefill_wall_ms"], 200.0)
        self.assertEqual(timing["pressure_peer_ready_after_count"], 1)
        self.assertEqual(timing["pressure_peer_ready_before_count"], 1)
        self.assertEqual(timing["pressure_peer_ready_inside_count"], 1)
        self.assertAlmostEqual(timing["pressure_window_duration_ms"], 2000.0)
        self.assertAlmostEqual(
            timing["pressure_window_start_after_peer_dispatch_ms"], 500.0
        )
        self.assertFalse(
            serve.healthy_peer_overlaps_pressure(
                result("actor-wide-pause", 1.5, 4.5, [1.9, 4.1]), pressure
            )
        )
        self.assertFalse(
            serve.healthy_peer_overlaps_pressure(
                result("active-before-timeout", 1.5, 4.5, [1.9, 3.9]), pressure
            )
        )
        self.assertFalse(
            serve.healthy_peer_overlaps_pressure(
                result("active-after-start", 1.5, 4.5, [2.1, 4.1]), pressure
            )
        )
        self.assertFalse(
            serve.healthy_peer_overlaps_pressure(
                result("after", 5.0, 6.0, [5.1]), pressure
            )
        )
        self.assertFalse(
            serve.healthy_peer_overlaps_pressure(
                result("pressure-peer", 1.5, 4.5, [1.9, 3.0, 4.1]), None
            )
        )

    def test_environment_sanitizer_keeps_only_runner_owned_kiln_controls(self) -> None:
        sanitized = serve.sanitized_environment(
            {
                "PATH": "/bin",
                "HOME": "/tmp",
                serve.RESULT_ENV: "/tmp/result.json",
                serve.VARIANT_ENV: "graphs-off",
                "RUST_LOG": "trace",
            }
        )
        self.assertEqual(sanitized, {"PATH": "/bin", "HOME": "/tmp"})

    def test_environment_sanitizer_rejects_ambient_server_controls(self) -> None:
        with self.assertRaisesRegex(
            serve.QualificationError,
            "KILN_MAX_DECODE_BATCH, KILN_MODEL_PATH, KILN_ROCM_GRAPHS",
        ):
            serve.sanitized_environment(
                {
                    "PATH": "/bin",
                    "KILN_MODEL_PATH": "wrong",
                    "KILN_ROCM_GRAPHS": "0",
                    "KILN_MAX_DECODE_BATCH": "12",
                }
            )

    def test_server_launch_uses_typed_file_and_only_internal_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            adapters = Path(tmp) / "adapters"
            snapshots = Path(tmp) / "snapshots"
            for variant, config in serve.VARIANT_CONFIGS.items():
                with self.subTest(variant=variant):
                    config_path = Path(tmp) / f"{variant}.toml"
                    serve.write_server_config(
                        config_path, variant, model, 1234, adapters, snapshots
                    )
                    parsed = parse_generated_toml(
                        config_path.read_text(encoding="utf-8")
                    )
                    expected = config["runtime"]
                    self.assertEqual(
                        parsed["memory"]["reclaim_mode"],
                        expected["memory_reclaim_requested_mode"],
                    )
                    self.assertEqual(
                        parsed["memory"]["kv_autoscale"],
                        expected["kv_autoscale_requested"],
                    )
                    self.assertEqual(parsed["memory"]["kv_force_blocks"], 0)
                    self.assertEqual(
                        parsed["server"]["serving_profile"],
                        expected["serving_profile"],
                    )
                    self.assertEqual(
                        parsed["server"]["stream_stall_grace_ms"],
                        serve.STREAM_STALL_GRACE_MS,
                    )
                    self.assertEqual(
                        parsed["server"]["http_send_buffer_bytes"],
                        serve.HTTP_SEND_BUFFER_BYTES,
                    )
                    self.assertTrue(parsed["server"]["chat_performance_metadata"])
                    self.assertFalse(parsed["server"]["default_thinking_enabled"])
                    self.assertEqual(parsed["model"]["path"], str(model))
                    self.assertEqual(parsed["model"]["model_id"], serve.MODEL_SOURCE_ID)
                    self.assertEqual(parsed["model"]["adapter_dir"], str(adapters))
                    self.assertEqual(parsed["model"]["snapshot_dir"], str(snapshots))
                    self.assertEqual(parsed["model"]["served_model_id"], serve.MODEL_ID)
                    self.assertEqual(
                        parsed["accelerator"]["rocm_graph_mode"],
                        "profile" if expected["rocm_graphs_requested"] else "disabled",
                    )

                    env = serve.server_environment(variant)
                    self.assertEqual(env["KILN_DEBUG_ENDPOINTS"], "1")
                    self.assertEqual(
                        sorted(key for key in env if key.startswith("KILN_")),
                        ["KILN_DEBUG_ENDPOINTS"],
                    )

    def test_server_config_overrides_are_typed_and_source_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "kiln.toml"
            serve.write_server_config(
                path,
                "default",
                root / 'model "quoted"',
                4321,
                root / "adapters",
                root / "snapshots",
                deterministic=True,
                rocm_synchronization_mode="stream_ordered",
                rocm_graph_mode="disabled",
                rocm_graph_cache_entries=12,
                rocm_graph_cache_max_bytes=64 << 20,
                kv_force_blocks=7,
            )
            parsed = parse_generated_toml(path.read_text(encoding="utf-8"))

        self.assertTrue(parsed["server"]["deterministic"])
        self.assertEqual(parsed["server"]["port"], 4321)
        self.assertEqual(
            parsed["accelerator"]["rocm_synchronization_mode"],
            "stream_ordered",
        )
        self.assertEqual(parsed["accelerator"]["rocm_graph_mode"], "disabled")
        self.assertEqual(parsed["accelerator"]["rocm_graph_cache_entries"], 12)
        self.assertEqual(
            parsed["accelerator"]["rocm_graph_cache_max_bytes"], 64 << 20
        )
        self.assertEqual(parsed["memory"]["kv_force_blocks"], 7)
        self.assertEqual(parsed["model"]["path"], str(root / 'model "quoted"'))

    def test_server_launcher_requires_and_consumes_one_typed_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "kiln"
            config = root / "kiln.toml"
            binary.write_bytes(b"fixture")
            config.write_text("[server]\nport = 8420\n")
            process = mock.Mock(stdout=object())
            server_log = mock.Mock()
            environment = {"RUST_LOG": "kiln=info"}
            with (
                mock.patch.object(serve, "server_environment", return_value=environment),
                mock.patch.object(serve.subprocess, "Popen", return_value=process) as popen,
                mock.patch.object(serve, "ServerLog", return_value=server_log) as log_type,
            ):
                observed_process, observed_log = serve.start_server(
                    binary, config, "default"
                )

            self.assertIs(observed_process, process)
            self.assertIs(observed_log, server_log)
            popen.assert_called_once_with(
                [str(binary), "--config", str(config), "serve"],
                cwd=serve.ROOT,
                env=environment,
                stdout=serve.subprocess.PIPE,
                stderr=serve.subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            log_type.assert_called_once_with(process.stdout)
            server_log.start.assert_called_once_with()

            with self.assertRaisesRegex(
                serve.QualificationError, "configuration is missing"
            ):
                serve.start_server(binary, root / "missing.toml", "default")

    def test_server_termination_reports_graceful_and_forced_outcomes(self) -> None:
        graceful = mock.Mock(pid=101)
        graceful.poll.return_value = None
        graceful.wait.return_value = 0
        with mock.patch.object(serve.os, "killpg") as killpg:
            outcome = serve.terminate_process(graceful)
        self.assertEqual(outcome.returncode, 0)
        self.assertFalse(outcome.forced)
        killpg.assert_called_once_with(101, serve.signal.SIGTERM)
        graceful.wait.assert_called_once_with(
            timeout=serve.SERVER_SHUTDOWN_GRACE_SECONDS
        )

        forced = mock.Mock(pid=202)
        forced.poll.return_value = None
        forced.wait.side_effect = [
            subprocess.TimeoutExpired("kiln", serve.SERVER_SHUTDOWN_GRACE_SECONDS),
            -serve.signal.SIGKILL,
        ]
        with mock.patch.object(serve.os, "killpg") as killpg:
            outcome = serve.terminate_process(forced)
        self.assertEqual(outcome.returncode, -serve.signal.SIGKILL)
        self.assertTrue(outcome.forced)
        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(202, serve.signal.SIGTERM),
                mock.call(202, serve.signal.SIGKILL),
            ],
        )

    def test_snapshot_residue_ignores_empty_tempdirs_but_not_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / ".kiln-model-snapshot-test"
            snapshot.mkdir()
            self.assertEqual(serve.snapshot_payload_residue(root), [])

            payload = snapshot / "model.safetensors"
            payload.write_bytes(b"weights")
            self.assertEqual(
                serve.snapshot_payload_residue(root),
                [".kiln-model-snapshot-test/model.safetensors"],
            )

    def test_http_send_buffer_attestation_is_platform_strict(self) -> None:
        linux = {
            "send_buffer_requested_bytes": 4096,
            "send_buffer_kernel_readback_bytes": 16384,
            "send_buffer_effective_bytes": 8192,
        }
        self.assertEqual(
            serve.http_send_buffer_attestation_failures(
                linux,
                label="health",
                expected_requested_bytes=4096,
                platform_name="linux",
            ),
            [],
        )
        non_linux = dict(linux)
        non_linux["send_buffer_kernel_readback_bytes"] = 8192
        self.assertEqual(
            serve.http_send_buffer_attestation_failures(
                non_linux,
                label="debug",
                expected_requested_bytes=4096,
                platform_name="darwin",
            ),
            [],
        )

        invalid_cases = {
            "requested": {**linux, "send_buffer_requested_bytes": 8192},
            "raw-zero": {**linux, "send_buffer_kernel_readback_bytes": 0},
            "effective-small": {
                **linux,
                "send_buffer_kernel_readback_bytes": 4096,
                "send_buffer_effective_bytes": 2048,
            },
            "relationship": {**linux, "send_buffer_kernel_readback_bytes": 8192},
        }
        for name, value in invalid_cases.items():
            with self.subTest(name=name):
                self.assertTrue(
                    serve.http_send_buffer_attestation_failures(
                        value,
                        label="health",
                        expected_requested_bytes=4096,
                        platform_name="linux",
                    )
                )

    def test_gpu_memory_attestation_requires_live_rocm_evidence(self) -> None:
        valid = {
            "total_vram_bytes": 1024,
            "live": {"used_gb": 0.0, "source": "linux-drm-sysfs"},
        }
        self.assertEqual(serve.gpu_memory_attestation_failures(valid), [])

        invalid_cases = (
            None,
            {},
            {**valid, "total_vram_bytes": True},
            {**valid, "total_vram_bytes": 0},
            {**valid, "live": None},
            {**valid, "live": {"used_gb": -1.0, "source": "probe"}},
            {**valid, "live": {"used_gb": float("nan"), "source": "probe"}},
            {**valid, "live": {"used_gb": False, "source": "probe"}},
            {**valid, "live": {"used_gb": 1.0, "source": "  "}},
        )
        for value in invalid_cases:
            with self.subTest(value=value):
                self.assertTrue(serve.gpu_memory_attestation_failures(value))

    def test_runtime_attestation_accepts_every_declared_variant(self) -> None:
        for variant, config in serve.VARIANT_CONFIGS.items():
            runtime = config["runtime"]
            with self.subTest(variant=variant):
                failures = serve.attest_runtime(
                    variant,
                    health_fixture(
                        kv_autoscale=runtime["kv_autoscale_enabled"],
                        rocm_graphs=runtime["rocm_graphs_enabled"],
                        serving_profile=runtime["serving_profile"],
                        kv_autoscale_requested=runtime["kv_autoscale_requested"],
                        rocm_graphs_requested=runtime["rocm_graphs_requested"],
                        memory_reclaim_requested_mode=runtime[
                            "memory_reclaim_requested_mode"
                        ],
                        memory_reclaim_mode=runtime["memory_reclaim_mode"],
                    ),
                    debug_fixture(
                        kv_autoscale=runtime["kv_autoscale_requested"],
                        rocm_graphs=runtime["rocm_graphs_requested"],
                        serving_profile=runtime["serving_profile"],
                        rocm_graphs_enabled=runtime["rocm_graphs_enabled"],
                        memory_reclaim_requested_mode=runtime[
                            "memory_reclaim_requested_mode"
                        ],
                    ),
                )
                self.assertEqual(failures, [])

    def test_runtime_attestation_rejects_effective_state_or_source_drift(self) -> None:
        health = health_fixture(kv_autoscale=True, rocm_graphs=True)
        debug = debug_fixture(kv_autoscale=True, rocm_graphs=True)
        health["decode_runtime"]["rocm_graphs"]["enabled"] = False
        health["decode_runtime"]["rocm_graphs"]["captured_graph_count"] = 9
        health["decode_runtime"]["rocm_graphs"]["retained_bytes"] = (1 << 30) + 1
        health["decode_runtime"]["rocm_graphs"][
            "retained_bytes_accounting_complete"
        ] = False
        health["decode_runtime"]["rocm_graphs"]["quarantined_retained_bytes"] = 4096
        debug["kv_autoscaler"]["requested_source"] = "environment"
        debug["http"]["send_buffer_effective_bytes"] *= 2
        debug["http"]["send_buffer_kernel_readback_bytes"] *= 2
        health["decode_runtime"]["batching_engine"][
            "stream_stall_grace_source"
        ] = "default"
        health["decode_runtime"]["batching_engine"]["max_active_requests"] = 99
        debug["batching_engine"]["snapshot"]["max_prefill_staging_slots"] = 0
        debug["env_flags"]["KILN_STREAM_STALL_GRACE_MS"]["value"] = "10"
        failures = serve.attest_runtime("default", health, debug)
        self.assertTrue(any("ROCm graph enabled" in failure for failure in failures))
        self.assertTrue(any("entry count exceeds" in failure for failure in failures))
        self.assertTrue(any("retained bytes exceed" in failure for failure in failures))
        self.assertTrue(any("accounting is incomplete" in failure for failure in failures))
        self.assertTrue(any("quarantined retained bytes" in failure for failure in failures))
        self.assertTrue(
            any("debug KV autoscaler requested_source" in failure for failure in failures)
        )
        self.assertTrue(any("disagree exactly" in failure for failure in failures))
        self.assertTrue(any("grace source" in failure for failure in failures))
        self.assertTrue(
            any(
                "stream-stall grace compatibility environment flag" in failure
                for failure in failures
            )
        )
        self.assertTrue(
            any("health batching max_active_requests" in failure for failure in failures)
        )
        self.assertTrue(
            any(
                "debug batching max_prefill_staging_slots" in failure
                for failure in failures
            )
        )

    def test_runtime_attestation_accepts_automatic_monitor_when_effective(self) -> None:
        variant = "test-automatic"
        serve.VARIANT_CONFIGS[variant] = serve._variant_config(
            serving_profile="experimental",
            kv_autoscale_requested=False,
            kv_autoscale_enabled=False,
            memory_reclaim_requested_mode="automatic",
            memory_reclaim_mode="automatic",
            rocm_graphs_requested=False,
            rocm_graphs_enabled=False,
        )
        try:
            failures = serve.attest_runtime(
                variant,
                health_fixture(
                    kv_autoscale=False,
                    rocm_graphs=False,
                    memory_reclaim_requested_mode="automatic",
                    memory_reclaim_mode="automatic",
                ),
                debug_fixture(
                    kv_autoscale=False,
                    rocm_graphs=False,
                    memory_reclaim_requested_mode="automatic",
                ),
            )
        finally:
            del serve.VARIANT_CONFIGS[variant]
        self.assertEqual(failures, [])

    def test_runtime_execution_requires_capture_and_replay_only_when_enabled(self) -> None:
        warmup = health_fixture(kv_autoscale=True, rocm_graphs=True)
        warmup_graph = warmup["decode_runtime"]["rocm_graphs"]
        warmup_graph.update(
            {
                "capture_attempts": 2,
                "capture_successes": 1,
                "capture_deferrals": 1,
                "replay_attempts": 1,
                "replay_successes": 1,
                "cache_admission_successes": 1,
                "captured_graph_count": 1,
                "opaque_native_object_count": 5,
                "graph_slot_create_count": 1,
                "graph_slot_count": 1,
                "idle_graph_slot_count": 1,
                "last_transient_candidate_bytes": 100_000_000,
                "peak_transient_candidate_bytes": 123_000_000,
                "candidate_warm_phase": {
                    "calls": 2,
                    "slow": 1,
                    "total_duration_micros": 150_000,
                    "max_duration_micros": 120_000,
                },
                "pre_native_reservation_phase": {
                    "calls": 1,
                    "slow": 0,
                    "total_duration_micros": 20_000,
                    "max_duration_micros": 20_000,
                },
                "native_capture_phase": {
                    "calls": 1,
                    "slow": 1,
                    "total_duration_micros": 130_000,
                    "max_duration_micros": 130_000,
                },
            }
        )
        after = json.loads(json.dumps(warmup))
        after_graph = after["decode_runtime"]["rocm_graphs"]
        after_graph["replay_attempts"] = 8
        after_graph["replay_successes"] = 8
        self.assertEqual(serve.attest_runtime_execution("default", warmup, after), [])

        after_graph["replay_attempts"] = 1
        after_graph["replay_successes"] = 1
        failures = serve.attest_runtime_execution("default", warmup, after)
        self.assertTrue(any("measured graph-on load" in failure for failure in failures))

        graphs_off = health_fixture(kv_autoscale=True, rocm_graphs=False)
        self.assertEqual(
            serve.attest_runtime_execution("graphs-off", graphs_off, graphs_off), []
        )
        graphs_off["decode_runtime"]["rocm_graphs"].update(
            {
                "capture_attempts": 1,
                "capture_successes": 1,
                "cache_admission_successes": 1,
                "captured_graph_count": 1,
                "opaque_native_object_count": 5,
                "graph_slot_create_count": 1,
                "graph_slot_count": 1,
                "idle_graph_slot_count": 1,
            }
        )
        failures = serve.attest_runtime_execution("graphs-off", graphs_off, graphs_off)
        self.assertTrue(any("capture_attempts=1" in failure for failure in failures))

    def test_disabled_policy_gates_cover_startup_events_and_block_count(self) -> None:
        startup_events = [
            serve.ObservedEvent(1.0, "kv_resize", "resize during warmup"),
            serve.ObservedEvent(1.1, "memory_reclaim", "reclaim during startup"),
        ]
        failures = serve.disabled_policy_attestation_failures(
            "autoscale-off",
            startup_events,
            initial_blocks_total=100,
            final_blocks_total=101,
        )
        self.assertTrue(any("reclaim event" in failure for failure in failures))
        self.assertTrue(any("resize event" in failure for failure in failures))
        self.assertTrue(any("blocks_total" in failure for failure in failures))

        graph_failures = serve.disabled_policy_attestation_failures(
            "graphs-off",
            [
                serve.ObservedEvent(1.0, "graph_capture", "warmup capture"),
                serve.ObservedEvent(1.1, "graph_sync", "warmup synchronization"),
            ],
            initial_blocks_total=100,
            final_blocks_total=100,
        )
        self.assertTrue(any("graphs-off" in failure for failure in graph_failures))
        self.assertEqual(
            serve.disabled_policy_attestation_failures(
                "both-off",
                [],
                initial_blocks_total=100,
                final_blocks_total=100,
            ),
            [],
        )

    def test_graph_snapshot_rejects_missing_null_and_inconsistent_counters(self) -> None:
        health = health_fixture(kv_autoscale=True, rocm_graphs=False)
        del health["decode_runtime"]["rocm_graphs"]["capture_attempts"]
        with self.assertRaises(serve.QualificationError):
            serve.graph_snapshot(health)

        health = health_fixture(kv_autoscale=True, rocm_graphs=False)
        health["decode_runtime"]["rocm_graphs"]["replay_attempts"] = None
        with self.assertRaises(serve.QualificationError):
            serve.graph_snapshot(health)

        health = health_fixture(kv_autoscale=True, rocm_graphs=False)
        health["decode_runtime"]["rocm_graphs"]["capture_attempts"] = 1
        with self.assertRaises(serve.QualificationError):
            serve.graph_snapshot(health)

        health = health_fixture(kv_autoscale=True, rocm_graphs=True)
        del health["decode_runtime"]["rocm_graphs"]["fallbacks"]
        with self.assertRaisesRegex(serve.QualificationError, "fallbacks is missing"):
            serve.graph_snapshot(health)

        health = health_fixture(kv_autoscale=True, rocm_graphs=True)
        health["decode_runtime"]["rocm_graphs"]["fallbacks"].update(
            {
                "total": 1,
                "replay_failure": 1,
                "slow": 1,
                "total_duration_micros": 120_000,
                "max_duration_micros": 120_000,
            }
        )
        snapshot = serve.graph_snapshot(health)
        self.assertEqual(snapshot["fallback_total"], 1)
        self.assertEqual(snapshot["fallback_replay_failure"], 1)

        inconsistent = json.loads(json.dumps(health))
        inconsistent["decode_runtime"]["rocm_graphs"]["fallbacks"]["total"] = 2
        with self.assertRaisesRegex(serve.QualificationError, "do not sum"):
            serve.graph_snapshot(inconsistent)

        inconsistent = json.loads(json.dumps(health))
        inconsistent["decode_runtime"]["rocm_graphs"]["fallbacks"]["slow"] = 2
        with self.assertRaisesRegex(serve.QualificationError, "slow fallback"):
            serve.graph_snapshot(inconsistent)

        inconsistent = json.loads(json.dumps(health))
        inconsistent["decode_runtime"]["rocm_graphs"]["fallbacks"][
            "max_duration_micros"
        ] = 120_001
        with self.assertRaisesRegex(serve.QualificationError, "max fallback"):
            serve.graph_snapshot(inconsistent)

    def test_external_yield_sync_snapshot_is_strict_and_reports_deltas(self) -> None:
        before = health_fixture(kv_autoscale=True, rocm_graphs=True)
        before["backend_runtime"]["external_yield_sync"] = [
            {
                "boundary": "batched decode step",
                "calls": 2,
                "failures": 0,
                "total_micros": 20_000,
                "max_micros": 12_000,
                "slow_calls": 0,
            }
        ]
        after = json.loads(json.dumps(before))
        after["backend_runtime"]["external_yield_sync"][0].update(
            {
                "calls": 7,
                "total_micros": 95_000,
                "max_micros": 25_000,
            }
        )
        self.assertEqual(
            serve.external_yield_sync_metric_values(before, after),
            {
                "external_yield_sync_call_count": 5,
                "external_yield_sync_failure_count": 0,
                "external_yield_sync_max_ms": 25.0,
                "external_yield_sync_slow_count": 0,
                "external_yield_sync_total_ms": 75.0,
            },
        )

        duplicate = json.loads(json.dumps(after))
        duplicate["backend_runtime"]["external_yield_sync"].append(
            duplicate["backend_runtime"]["external_yield_sync"][0]
        )
        with self.assertRaisesRegex(serve.QualificationError, "duplicate"):
            serve.external_yield_sync_snapshot(duplicate)

        regressed = json.loads(json.dumps(after))
        regressed["backend_runtime"]["external_yield_sync"][0]["calls"] = 1
        with self.assertRaisesRegex(serve.QualificationError, "regressed"):
            serve.external_yield_sync_metric_values(before, regressed)

    def test_cancellation_match_requires_marker_and_disconnect_reason(self) -> None:
        records = [
            {"prompt_preview": "marker-a", "finish_reason": "stop"},
            {"prompt_preview": "marker-b", "finish_reason": "client_disconnect"},
        ]
        self.assertTrue(serve.cancellation_recorded(records, "marker-b"))
        self.assertFalse(serve.cancellation_recorded(records, "marker-a"))
        drained = {
            "active_decode": 0,
            "active_prefill": 0,
            "active_staged_requests": 0,
            "queue_depth": 0,
        }
        self.assertFalse(
            serve.batching_engine_drained({**drained, "active_decode": 1})
        )
        self.assertFalse(
            serve.batching_engine_drained({**drained, "active_prefill": 1})
        )
        self.assertFalse(
            serve.batching_engine_drained({**drained, "active_staged_requests": 1})
        )
        self.assertFalse(serve.batching_engine_drained({**drained, "queue_depth": 1}))
        self.assertTrue(serve.batching_engine_drained(drained))
        for malformed in (
            None,
            {},
            {"active_decode": 0},
            {**drained, "active_decode": "0"},
            {**drained, "active_decode": False},
            {**drained, "active_decode": -1},
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(serve.QualificationError):
                    serve.batching_engine_drained(malformed)

    def test_outlier_attribution_uses_only_events_inside_gap(self) -> None:
        result = serve.StreamResult(
            name="measured",
            marker="m",
            started=0.0,
            finished=1.0,
            semantic_times=[0.0, 0.1, 0.2],
            token_ready_times=[0.0, 0.1, 0.7],
            token_queue_delays_ms=[0.0, 0.0, 0.0],
            prompt_tokens=1,
            completion_tokens=3,
            usage_records=1,
            finish_reason="stop",
            done=True,
            cancelled=False,
            error=None,
        )
        inside = serve.ObservedEvent(0.5, "memory_reclaim", "trim")
        outside = serve.ObservedEvent(0.8, "kv_resize", "resize")
        attributed, unexplained = serve.classify_itl_outliers(
            [100.0, 100.0], [result], [inside, outside]
        )
        self.assertEqual((attributed, unexplained), (1, 0))
        attributed, unexplained = serve.classify_itl_outliers(
            [100.0, 100.0], [result], [outside]
        )
        self.assertEqual((attributed, unexplained), (0, 1))
        synchronization = serve.ObservedEvent(
            0.5, "external_yield_sync", "slow attributed synchronization"
        )
        attributed, unexplained = serve.classify_itl_outliers(
            [100.0, 100.0], [result], [synchronization]
        )
        self.assertEqual((attributed, unexplained), (1, 0))
        prefill = serve.ObservedEvent(0.7, "actor_prefill", "slow prefill")
        attributed, unexplained = serve.classify_itl_outliers(
            [100.0, 100.0], [result], [prefill]
        )
        self.assertEqual((attributed, unexplained), (1, 0))

    def test_metric_values_use_runtime_counter_deltas(self) -> None:
        result = serve.StreamResult(
            name="long-prefill",
            marker="m",
            started=1.0,
            finished=1.5,
            semantic_times=[1.1, 1.2],
            token_ready_times=[1.1, 1.2],
            token_queue_delays_ms=[0.0, 0.0],
            prompt_tokens=100,
            completion_tokens=2,
            usage_records=1,
            finish_reason="length",
            done=True,
            cancelled=False,
            error=None,
        )
        warmup = serve.StreamResult(
            name="warmup",
            marker="w",
            started=0.0,
            finished=0.3,
            semantic_times=[0.1, 0.2],
            token_ready_times=[0.1, 0.2],
            token_queue_delays_ms=[0.0, 0.0],
            prompt_tokens=10,
            completion_tokens=2,
            usage_records=1,
            finish_reason="length",
            done=True,
            cancelled=False,
            error=None,
        )
        before = health_fixture(kv_autoscale=True, rocm_graphs=True)
        measurement_start = json.loads(json.dumps(before))
        measurement_batching = measurement_start["decode_runtime"]["batching_engine"]
        measurement_batching.update(
            {
                "total_errors": 4,
                "total_decode_forwards": 10,
                "total_batched_decode_forwards": 2,
                "total_decode_rows": 20,
                "total_decode_forward_ms": 200.0,
                "max_decode_forward_ms": 50.0,
                "slow_decode_forward_count": 0,
                "total_prefill_forwards": 4,
                "total_prefill_forward_ms": 100.0,
                "max_prefill_forward_ms": 40.0,
                "slow_prefill_forward_count": 0,
                "total_admission_calls": 3,
                "total_admission_ms": 30.0,
                "max_admission_ms": 20.0,
                "slow_admission_count": 0,
                "response_backpressure_events": 3,
                "response_backpressure_wait_ms": 100,
                "response_stall_evictions": 2,
                "total_short_prefill_priority_forwards": 2,
                "total_prefill_staging_admissions": 1,
                "total_prefill_staging_priority_forwards": 1,
                "max_observed_active_requests": serve.MAX_DECODE_BATCH,
            }
        )
        measurement_start["backend_runtime"]["external_yield_sync"] = [
            {
                "boundary": "batched decode step",
                "calls": 2,
                "failures": 0,
                "total_micros": 20_000,
                "max_micros": 12_000,
                "slow_calls": 0,
            }
        ]
        end = json.loads(json.dumps(measurement_start))
        end["backend_runtime"]["external_yield_sync"][0].update(
            {
                "calls": 7,
                "total_micros": 95_000,
                "max_micros": 25_000,
            }
        )
        end_graph = end["decode_runtime"]["rocm_graphs"]
        end_graph.update(
            {
                "capture_attempts": 2,
                "capture_successes": 1,
                "capture_deferrals": 1,
                "replay_attempts": 8,
                "replay_successes": 8,
                "pre_candidate_headroom_phase": {
                    "calls": 2,
                    "slow": 0,
                    "total_duration_micros": 40_000,
                    "max_duration_micros": 25_000,
                },
                "cache_admission_successes": 1,
                "captured_graph_count": 1,
                "opaque_native_object_count": 5,
                "graph_slot_create_count": 1,
                "graph_slot_count": 1,
                "idle_graph_slot_count": 1,
                "last_transient_candidate_bytes": 100_000_000,
                "peak_transient_candidate_bytes": 123_000_000,
                "candidate_warm_phase": {
                    "calls": 2,
                    "slow": 1,
                    "total_duration_micros": 150_000,
                    "max_duration_micros": 120_000,
                },
                "pre_native_reservation_phase": {
                    "calls": 1,
                    "slow": 0,
                    "total_duration_micros": 20_000,
                    "max_duration_micros": 20_000,
                },
                "native_capture_phase": {
                    "calls": 1,
                    "slow": 1,
                    "total_duration_micros": 130_000,
                    "max_duration_micros": 130_000,
                },
            }
        )
        end_batching = end["decode_runtime"]["batching_engine"]
        end_batching.update(
            {
                "total_errors": 5,
                "total_decode_forwards": 15,
                "total_batched_decode_forwards": 6,
                "total_decode_rows": 35,
                "total_decode_forward_ms": 800.0,
                "max_decode_forward_ms": 150.0,
                "slow_decode_forward_count": 2,
                "total_prefill_forwards": 9,
                "total_prefill_forward_ms": 1_500.0,
                "max_prefill_forward_ms": 600.0,
                "slow_prefill_forward_count": 3,
                "total_admission_calls": 8,
                "total_admission_ms": 110.0,
                "max_admission_ms": 120.0,
                "slow_admission_count": 1,
                "response_backpressure_events": 5,
                "response_backpressure_wait_ms": 850,
                "response_stall_evictions": 3,
                "total_short_prefill_priority_forwards": 7,
                "total_prefill_staging_admissions": 4,
                "total_prefill_staging_priority_forwards": 5,
                "max_observed_active_requests": serve.MAX_ACTIVE_REQUESTS - 1,
            }
        )

        values = serve.metric_values(
            measured=[result],
            warmup=warmup,
            long_prefill=result,
            cancellation_confirmed=True,
            slow_peer_success=1,
            pressure_peer=result,
            pressure_window=serve.DeliveryPressureWindow(
                request_id="slow-id",
                client="qualification-slow-marker",
                started=1.15,
                timed_out=1.18,
            ),
            peak_memory=123,
            health_after_warmup=before,
            health_measurement_start=measurement_start,
            health_end=end,
            events=[],
        )
        self.assertEqual(values["client_backpressure_event_count"], 2)
        self.assertEqual(values["client_backpressure_wait_ms"], 750)
        self.assertEqual(values["client_stall_eviction_count"], 1)
        self.assertAlmostEqual(
            values["pressure_peer_first_ready_after_dispatch_ms"], 100.0
        )
        self.assertEqual(values["pressure_peer_ready_before_count"], 1)
        self.assertEqual(values["pressure_peer_ready_inside_count"], 0)
        self.assertEqual(values["pressure_peer_ready_after_count"], 1)
        self.assertAlmostEqual(values["pressure_window_duration_ms"], 30.0)
        self.assertAlmostEqual(
            values["pressure_window_start_after_peer_dispatch_ms"], 150.0
        )
        self.assertEqual(values["batching_total_errors"], 1)
        self.assertEqual(values["batching_decode_forward_count"], 5)
        self.assertEqual(values["batching_batched_decode_forward_count"], 4)
        self.assertEqual(values["batching_decode_row_count"], 15)
        self.assertEqual(values["batching_mean_rows_per_forward"], 3.0)
        self.assertEqual(values["batching_max_decode_batch"], serve.MAX_DECODE_BATCH)
        self.assertEqual(
            values["batching_prefill_staging_slot_count"],
            serve.MAX_PREFILL_STAGING_SLOTS,
        )
        self.assertEqual(
            values["batching_max_active_requests"], serve.MAX_ACTIVE_REQUESTS
        )
        self.assertEqual(
            values["batching_prefill_staging_priority_burst"],
            serve.MAX_PREFILL_STAGING_PRIORITY_BURST,
        )
        self.assertEqual(
            values["batching_max_observed_active_requests"],
            serve.MAX_ACTIVE_REQUESTS - 1,
        )
        self.assertEqual(values["batching_prefill_staging_admission_count"], 3)
        self.assertEqual(
            values["batching_prefill_staging_priority_forward_count"], 4
        )
        self.assertEqual(
            values["batching_max_prefill_tokens_per_cycle"],
            serve.MAX_PREFILL_TOKENS_PER_CYCLE,
        )
        self.assertEqual(
            values["batching_max_prefill_layers_per_cycle"],
            serve.MAX_PREFILL_LAYERS_PER_CYCLE,
        )
        self.assertEqual(values["batching_decode_forward_ms_total"], 600.0)
        self.assertEqual(values["batching_decode_forward_ms_max"], 150.0)
        self.assertEqual(values["batching_slow_decode_forward_count"], 2)
        self.assertEqual(values["batching_prefill_forward_count"], 5)
        self.assertEqual(values["batching_prefill_layer_count"], 0)
        self.assertEqual(values["batching_prefill_layer_yield_count"], 0)
        self.assertEqual(values["batching_short_prefill_priority_forward_count"], 5)
        self.assertEqual(values["batching_prefill_forward_ms_total"], 1_400.0)
        self.assertEqual(values["batching_prefill_forward_ms_max"], 600.0)
        self.assertEqual(values["batching_slow_prefill_forward_count"], 3)
        self.assertEqual(values["batching_admission_call_count"], 5)
        self.assertEqual(values["batching_admission_ms_total"], 80.0)
        self.assertEqual(values["batching_admission_ms_max"], 120.0)
        self.assertEqual(values["batching_slow_admission_count"], 1)
        self.assertEqual(values["graph_measured_capture_success_count"], 1)
        self.assertEqual(values["graph_measured_capture_deferral_count"], 1)
        self.assertEqual(values["graph_measured_replay_success_count"], 8)
        self.assertEqual(values["graph_measured_live_count_end"], 1)
        self.assertEqual(values["graph_candidate_warm_call_count"], 2)
        self.assertEqual(values["graph_candidate_warm_slow_count"], 1)
        self.assertEqual(values["graph_candidate_warm_duration_ms_total"], 150.0)
        self.assertEqual(values["graph_candidate_warm_duration_ms_max_end"], 120.0)
        self.assertEqual(values["graph_pre_native_reservation_call_count"], 1)
        self.assertEqual(values["graph_pre_candidate_headroom_call_count"], 2)
        self.assertEqual(values["graph_pre_candidate_headroom_duration_ms_total"], 40.0)
        self.assertEqual(values["graph_native_capture_slow_count"], 1)
        self.assertEqual(values["graph_rejected_candidate_cleanup_call_count"], 0)
        self.assertEqual(values["graph_transient_candidate_bytes_peak_end"], 123_000_000)
        self.assertEqual(values["length_terminated_request_count"], 1)
        self.assertEqual(values["external_yield_sync_call_count"], 5)
        self.assertEqual(values["external_yield_sync_failure_count"], 0)
        self.assertEqual(values["external_yield_sync_total_ms"], 75.0)
        self.assertEqual(values["external_yield_sync_max_ms"], 25.0)

        with self.assertRaises(serve.QualificationError):
            serve.counter_delta({"counter": 2}, {"counter": 1}, "counter")

    def test_staging_contract_requires_exact_capacity_and_measured_execution(self) -> None:
        good = {
            "batching_max_decode_batch": serve.MAX_DECODE_BATCH,
            "batching_prefill_staging_slot_count": serve.MAX_PREFILL_STAGING_SLOTS,
            "batching_max_active_requests": serve.MAX_ACTIVE_REQUESTS,
            "batching_prefill_staging_priority_burst": (
                serve.MAX_PREFILL_STAGING_PRIORITY_BURST
            ),
            "batching_max_observed_active_requests": serve.MAX_DECODE_BATCH + 1,
            "batching_prefill_staging_admission_count": 1,
            "batching_prefill_staging_priority_forward_count": 1,
            "batching_short_prefill_priority_forward_count": 2,
        }
        self.assertEqual(serve.batching_staging_contract_failures(good), [])

        mutations = {
            "batching_max_decode_batch": serve.MAX_DECODE_BATCH + 1,
            "batching_prefill_staging_slot_count": 0,
            "batching_max_active_requests": serve.MAX_ACTIVE_REQUESTS + 1,
            "batching_prefill_staging_priority_burst": 0,
            "batching_max_observed_active_requests": serve.MAX_DECODE_BATCH,
            "batching_prefill_staging_admission_count": 0,
            "batching_prefill_staging_priority_forward_count": 0,
        }
        for name, value in mutations.items():
            with self.subTest(name=name):
                failures = serve.batching_staging_contract_failures(
                    {**good, name: value}
                )
                self.assertTrue(failures)

        too_wide = {
            **good,
            "batching_max_observed_active_requests": serve.MAX_ACTIVE_REQUESTS + 1,
        }
        self.assertTrue(serve.batching_staging_contract_failures(too_wide))
        invalid_subset = {
            **good,
            "batching_prefill_staging_priority_forward_count": 3,
            "batching_short_prefill_priority_forward_count": 2,
        }
        self.assertTrue(serve.batching_staging_contract_failures(invalid_subset))

    def test_metric_contract_is_sorted_closed_and_finite(self) -> None:
        metrics = serve.zero_metrics()
        self.assertEqual(
            [metric["name"] for metric in metrics], sorted(serve.METRIC_DEFINITIONS)
        )
        self.assertEqual({metric["name"] for metric in metrics}, set(serve.METRIC_DEFINITIONS))
        self.assertEqual(
            next(m for m in metrics if m["name"] == "request_failure_count")["value"],
            1,
        )
        with self.assertRaises(serve.QualificationError):
            serve.metrics_from_values({"unknown": 1})

    def test_atomic_result_write_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            value = {"status": "passed"}
            serve.write_result(path, value)
            self.assertEqual(json.loads(path.read_text()), value)
            with self.assertRaises(FileExistsError):
                serve.write_result(path, value)
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_bounded_details_respects_case_result_limit(self) -> None:
        self.assertEqual(serve.bounded_details("short"), "short")
        bounded = serve.bounded_details("x" * 3000)
        assert bounded is not None
        self.assertLessEqual(len(bounded), 2000)
        self.assertTrue(bounded.endswith("...[details truncated]"))

    def test_request_body_pins_deterministic_non_thinking_stream(self) -> None:
        body = serve.request_body("prompt", 12, 7)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["seed"], 7)
        self.assertEqual(body["max_tokens"], 12)
        self.assertTrue(body["ignore_eos"])
        self.assertTrue(body["stream_options"]["include_usage"])
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        self.assertIsNone(body["adapter"])

    def test_request_body_can_pin_a_named_adapter(self) -> None:
        body = serve.request_body("prompt", 12, 7, adapter="qualification-adapter")
        self.assertEqual(body["adapter"], "qualification-adapter")

    def test_workload_markers_are_variant_invariant_and_path_safe(self) -> None:
        markers = {
            variant: serve.workload_marker(20260709, "normal-00")
            for variant in serve.VARIANT_CONFIGS
        }
        self.assertEqual(set(markers.values()), {"QUAL-20260709-normal-00"})
        self.assertTrue(
            all(
                config["workload"]["prompt_identity"] == serve.PROMPT_IDENTITY
                for config in serve.VARIANT_CONFIGS.values()
            )
        )
        with self.assertRaisesRegex(serve.QualificationError, "marker role"):
            serve.workload_marker(20260709, "../../default")
        with self.assertRaisesRegex(serve.QualificationError, "marker seed"):
            serve.workload_marker(-1, "normal-00")

    def test_measured_prompts_and_results_require_a_fixed_output_denominator(self) -> None:
        prompt = serve.deterministic_prompt("marker", 3)
        self.assertIn("ascending zero-padded integers", prompt)
        self.assertIn("until the response token limit", prompt)
        self.assertIn("item00 item01 item02", prompt)

        def completed(name: str, token_limit: int) -> serve.StreamResult:
            return serve.StreamResult(
                name=name,
                marker="marker",
                started=0.0,
                finished=1.0,
                semantic_times=[0.5],
                token_ready_times=[0.5] * token_limit,
                token_queue_delays_ms=[0.0] * token_limit,
                prompt_tokens=1,
                completion_tokens=token_limit,
                usage_records=1,
                finish_reason="length",
                done=True,
                cancelled=False,
                error=None,
            )

        measured = [
            completed(f"normal-{index:02d}", serve.NORMAL_MAX_TOKENS)
            for index in range(serve.NORMAL_REQUESTS)
        ]
        measured.extend(
            [
                completed("long-prefill", serve.LONG_PREFILL_MAX_TOKENS),
                completed("pressure-peer", serve.PRESSURE_PEER_MAX_TOKENS),
            ]
        )
        self.assertEqual(serve.fixed_output_contract_failures(measured), [])
        self.assertEqual(
            sum(result.completion_tokens for result in measured),
            serve.MEASURED_EXPECTED_COMPLETION_TOKENS,
        )

        measured[0] = serve.dataclasses.replace(
            measured[0],
            completion_tokens=serve.NORMAL_MAX_TOKENS - 1,
            token_ready_times=measured[0].token_ready_times[:-1],
            token_queue_delays_ms=measured[0].token_queue_delays_ms[:-1],
            finish_reason="stop",
        )
        failures = serve.fixed_output_contract_failures(measured)
        self.assertTrue(any("normal-00 must finish by length" in item for item in failures))
        self.assertTrue(any("completion total" in item for item in failures))

    def test_slow_consumer_prompt_demands_generation_until_the_token_limit(self) -> None:
        prompt = serve.slow_consumer_prompt("marker")
        self.assertIn("marker", prompt)
        self.assertIn("without commentary", prompt)
        self.assertIn("until the response token limit", prompt)

    def test_background_helpers_can_close_before_start(self) -> None:
        slow = serve.SlowConsumer(1, "marker", 7)
        slow.close()
        self.assertIsNone(slow.error)

        sampler = serve.MemorySampler(1)
        sampler.close()
        self.assertEqual(sampler.errors, [])


if __name__ == "__main__":
    unittest.main()
