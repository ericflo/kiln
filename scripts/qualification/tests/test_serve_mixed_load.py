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
    memory_reclaim_requested_mode: str = "off",
) -> dict:
    kv_autoscale_requested = (
        kv_autoscale
        if kv_autoscale_requested is None
        else kv_autoscale_requested
    )
    graph = {
        "requested": rocm_graphs,
        "capture_requested": rocm_graphs,
        "enabled": rocm_graphs,
        "capture_enabled": rocm_graphs,
        "state": "enabled" if rocm_graphs else "disabled",
        "capture_attempts": 0,
        "capture_successes": 0,
        "capture_deferrals": 0,
        "capture_failures": 0,
        "replay_attempts": 0,
        "replay_successes": 0,
        "replay_failures": 0,
        "failures": 0,
        "captured_graph_count": 0,
    }
    return {
        "backend": "model",
        "backend_runtime": {"external_yield_sync": []},
        "serving_profile": {
            "profile": serving_profile,
            "source": "environment",
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
            "rocm_graphs": graph,
            "kv_autoscaler": {
                "requested": kv_autoscale_requested,
                "enabled": kv_autoscale,
                "state": (
                    "unavailable"
                    if serving_profile == "stable"
                    else "enabled" if kv_autoscale else "disabled"
                ),
                "reason": (
                    "serving_profile_stable"
                    if serving_profile == "stable"
                    else "active" if kv_autoscale else "environment"
                ),
            },
            "memory_governor": {
                "reclaim_mode": "off",
                "requested_reclaim_mode": memory_reclaim_requested_mode,
                "automatic_monitor_enabled": False,
                "source": "environment",
                "disabled_by_serving_profile": serving_profile == "stable",
            },
            "batching_engine": {
                "stream_stall_grace_ms": serve.STREAM_STALL_GRACE_MS,
                "stream_stall_grace_source": "environment",
                "max_prefill_tokens_per_cycle": serve.MAX_PREFILL_TOKENS_PER_CYCLE,
                "max_prefill_tokens_per_cycle_source": "default",
                "max_prefill_layers_per_cycle": serve.MAX_PREFILL_LAYERS_PER_CYCLE,
                "max_prefill_layers_per_cycle_source": "default",
                "active_decode": 0,
                "queue_depth": 0,
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
                "total_prefill_token_budget_deferrals": 0,
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
    memory_reclaim_requested_mode: str = "off",
) -> dict:
    def flag(enabled: bool) -> dict:
        return {"present": not enabled, "value": None if enabled else "0"}

    return {
        "http": http_fixture(),
        "batching_engine": {
            "backend": "model",
            "enabled": True,
            "snapshot": {
                "stream_stall_grace_ms": serve.STREAM_STALL_GRACE_MS,
                "stream_stall_grace_source": "environment",
                "max_prefill_tokens_per_cycle": serve.MAX_PREFILL_TOKENS_PER_CYCLE,
                "max_prefill_tokens_per_cycle_source": "default",
                "max_prefill_layers_per_cycle": serve.MAX_PREFILL_LAYERS_PER_CYCLE,
                "max_prefill_layers_per_cycle_source": "default",
            },
        },
        "env_flags": {
            "KILN_KV_AUTOSCALE": flag(kv_autoscale),
            "KILN_ROCM_GRAPHS": flag(rocm_graphs),
            "KILN_MEMORY_RECLAIM_MODE": {
                "present": True,
                "value": memory_reclaim_requested_mode,
            },
            "KILN_HTTP_SEND_BUFFER_BYTES": {
                "present": True,
                "value": str(serve.HTTP_SEND_BUFFER_BYTES),
            },
            "KILN_STREAM_STALL_GRACE_MS": {
                "present": True,
                "value": str(serve.STREAM_STALL_GRACE_MS),
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
        response.getheader.return_value = "text/event-stream"
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
        sock.setblocking.assert_called_once_with(False)
        sock.settimeout.assert_called_once()
        response.read1.assert_called_once_with(4096)
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
            "features": "rocm",
            "locked": True,
            "no_default_features": True,
            "offline": True,
            "package": "kiln-server",
            "profile": "release",
            "rocm_archs": "gfx1151",
            "rocm_path": "/opt/rocm",
        }
        expected_schedule = {
            "cancellation_after_semantic_deltas": serve.CANCELLATION_AFTER_DELTAS,
            "long_prefill_max_tokens": serve.LONG_PREFILL_MAX_TOKENS,
            "long_prefill_words": serve.LONG_PREFILL_WORDS,
            "max_warmup_requests": serve.MAX_WARMUP_REQUESTS,
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
                "cargo",
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
            "token_index": 1,
            "ready_ms": 12.0,
            "handler_received_ms": 17.0,
            "queue_delay_ms": 5.0,
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
        timing["token_index"] = 1
        self.assertEqual(
            serve.parse_token_timing(timing, 1, previous_ready_ms=12.0),
            (12.0, 5.0),
        )
        with self.assertRaises(serve.QualificationError):
            serve.parse_token_timing(timing, 1, previous_ready_ms=13.0)

    def test_token_timing_usage_contract_allows_only_consumed_eos_delta(self) -> None:
        self.assertTrue(serve.token_timing_matches_usage("length", 3, 3))
        self.assertFalse(serve.token_timing_matches_usage("length", 2, 3))

        self.assertTrue(serve.token_timing_matches_usage("stop", 3, 3))
        self.assertTrue(serve.token_timing_matches_usage("stop", 3, 4))
        self.assertFalse(serve.token_timing_matches_usage("stop", 2, 4))
        self.assertFalse(serve.token_timing_matches_usage("stop", 0, 1))
        self.assertFalse(serve.token_timing_matches_usage(None, 3, 3))

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
            "KV autoscaler resized cache": "kv_resize",
            "ROCm pool reclaim completed": "memory_reclaim",
            "ROCm HIP graph captured for decode (24 layers)": "graph_capture",
            "ROCm graph capture failed: bad launch": "graph_fallback",
            "slow_backend_external_yield_sync": "external_yield_sync",
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

    def test_environment_sanitizer_removes_all_kiln_and_log_overrides(self) -> None:
        sanitized = serve.sanitized_environment(
            {
                "PATH": "/bin",
                "HOME": "/tmp",
                "KILN_MODEL_PATH": "wrong",
                "KILN_ROCM_GRAPHS": "0",
                "KILN_CONFIG": "wrong.toml",
                "RUST_LOG": "trace",
            }
        )
        self.assertEqual(sanitized, {"PATH": "/bin", "HOME": "/tmp"})

    def test_server_environment_uses_absence_for_defaults_and_zero_for_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            adapters = Path(tmp) / "adapters"
            snapshots = Path(tmp) / "snapshots"
            for variant, config in serve.VARIANT_CONFIGS.items():
                with self.subTest(variant=variant):
                    env = serve.server_environment(
                        variant, model, 1234, adapters, snapshots
                    )
                    expected = config["runtime"]
                    self.assertEqual(env["KILN_MODEL_SNAPSHOT_DIR"], str(snapshots))
                    self.assertEqual(
                        env["KILN_MEMORY_RECLAIM_MODE"],
                        expected["memory_reclaim_requested_mode"],
                    )
                    self.assertEqual(
                        env["KILN_SERVING_PROFILE"], expected["serving_profile"]
                    )
                    self.assertEqual(
                        env["KILN_STREAM_STALL_GRACE_MS"], str(serve.STREAM_STALL_GRACE_MS)
                    )
                    self.assertEqual(
                        env["KILN_HTTP_SEND_BUFFER_BYTES"],
                        str(serve.HTTP_SEND_BUFFER_BYTES),
                    )
                    self.assertEqual(env["KILN_CHAT_PERFORMANCE_METADATA"], "true")
                    self.assertEqual(env["KILN_DEBUG_ENDPOINTS"], "1")
                    self.assertEqual(env["KILN_LOG_FORMAT"], "json")
                    self.assertEqual(
                        env.get("KILN_KV_AUTOSCALE"),
                        None if expected["kv_autoscale_requested"] else "0",
                    )
                    self.assertEqual(
                        env.get("KILN_ROCM_GRAPHS"),
                        None if expected["rocm_graphs_requested"] else "0",
                    )

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
                        memory_reclaim_requested_mode=runtime[
                            "memory_reclaim_requested_mode"
                        ],
                    ),
                    debug_fixture(
                        kv_autoscale=runtime["kv_autoscale_requested"],
                        rocm_graphs=runtime["rocm_graphs_requested"],
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
        debug["env_flags"]["KILN_KV_AUTOSCALE"] = {"present": True, "value": "1"}
        debug["http"]["send_buffer_effective_bytes"] *= 2
        debug["http"]["send_buffer_kernel_readback_bytes"] *= 2
        health["decode_runtime"]["batching_engine"][
            "stream_stall_grace_source"
        ] = "default"
        debug["env_flags"]["KILN_STREAM_STALL_GRACE_MS"]["value"] = "10"
        failures = serve.attest_runtime("default", health, debug)
        self.assertTrue(any("ROCm graph enabled" in failure for failure in failures))
        self.assertTrue(any("must remain absent" in failure for failure in failures))
        self.assertTrue(any("disagree exactly" in failure for failure in failures))
        self.assertTrue(any("grace source" in failure for failure in failures))
        self.assertTrue(any("grace debug flag" in failure for failure in failures))

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
                "captured_graph_count": 1,
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
            {"capture_attempts": 1, "capture_successes": 1}
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
            [serve.ObservedEvent(1.0, "graph_capture", "warmup capture")],
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
        self.assertFalse(
            serve.batching_engine_drained({"active_decode": 1, "queue_depth": 0})
        )
        self.assertFalse(
            serve.batching_engine_drained({"active_decode": 0, "queue_depth": 1})
        )
        self.assertTrue(
            serve.batching_engine_drained({"active_decode": 0, "queue_depth": 0})
        )
        for malformed in (
            None,
            {},
            {"active_decode": 0},
            {"active_decode": "0", "queue_depth": 0},
            {"active_decode": False, "queue_depth": 0},
            {"active_decode": -1, "queue_depth": 0},
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
                "total_prefill_token_budget_deferrals": 2,
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
                "captured_graph_count": 1,
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
                "total_prefill_token_budget_deferrals": 7,
            }
        )

        values = serve.metric_values(
            measured=[result],
            warmup=warmup,
            long_prefill=result,
            cancellation_confirmed=True,
            slow_peer_success=1,
            peak_memory=123,
            health_after_warmup=before,
            health_measurement_start=measurement_start,
            health_end=end,
            events=[],
        )
        self.assertEqual(values["client_backpressure_event_count"], 2)
        self.assertEqual(values["client_backpressure_wait_ms"], 750)
        self.assertEqual(values["client_stall_eviction_count"], 1)
        self.assertEqual(values["batching_total_errors"], 1)
        self.assertEqual(values["batching_decode_forward_count"], 5)
        self.assertEqual(values["batching_batched_decode_forward_count"], 4)
        self.assertEqual(values["batching_decode_row_count"], 15)
        self.assertEqual(values["batching_mean_rows_per_forward"], 3.0)
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
        self.assertEqual(values["batching_prefill_token_budget_deferral_count"], 5)
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
        self.assertEqual(values["external_yield_sync_call_count"], 5)
        self.assertEqual(values["external_yield_sync_failure_count"], 0)
        self.assertEqual(values["external_yield_sync_total_ms"], 75.0)
        self.assertEqual(values["external_yield_sync_max_ms"], 25.0)

        with self.assertRaises(serve.QualificationError):
            serve.counter_delta({"counter": 2}, {"counter": 1}, "counter")

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
        self.assertTrue(body["stream_options"]["include_usage"])
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])

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
