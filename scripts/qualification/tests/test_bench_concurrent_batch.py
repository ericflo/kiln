from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack, redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "bench-concurrent-batch.py"
SPEC = importlib.util.spec_from_file_location("bench_concurrent_batch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def valid_vllm_manifest(model: str = "test-model") -> dict:
    identity = {
        "schema": "kiln.teacher-identity.v1",
        "served_model_id": model,
        "implementation": "vllm:0.25.0",
    }
    canonical_json = json.dumps(
        identity, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    payload = canonical_json.encode()
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return {
        "identity": identity,
        "canonical_json": canonical_json,
        "system_fingerprint": (
            f"kiln-teacher-v1.{encoded}.{hashlib.sha256(payload).hexdigest()}"
        ),
        "runtime_content_sha256": "a" * 64,
    }


def valid_host_thermal_policy() -> dict:
    return {
        "schema": bench.HOST_THERMAL_POLICY_SCHEMA,
        "id": "test-host-policy-v1",
        "sensor": {"hwmon_name": "fixture", "label": "package"},
        "limit_millicelsius": 90_000,
        "poll_interval_ms": 250,
        "pacing": {
            "start_millicelsius": 78_000,
            "resume_millicelsius": 70_000,
            "resume_stable_samples": 2,
        },
        "safe_handoff": {
            "target_millicelsius": 65_000,
            "stable_samples": 2,
            "timeout_seconds": 30.0,
        },
        "phase_settlement_timeout_seconds": 30.0,
    }


class FakeAttachedProcessGroup:
    pid = 4321

    def poll(self) -> None:
        return None

    def receipt_identity(self) -> dict:
        return {
            "pid": self.pid,
            "process_group_id": self.pid,
            "start_time_ticks": 123456,
            "boot_id": "fixture-boot-id",
            "executable": "/fixture/kiln-server",
            "cmdline_sha256": "sha256:" + "a" * 64,
        }


class FakeThermalGuard:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.trip_reason = None
        self.errors: list[str] = []
        self.samples = [50_000]
        self.input_path = Path("/fixture/temp1_input")
        self.phase = "startup"

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def start(self) -> None:
        return

    def close(self) -> None:
        return

    def wait_for_pacing_settlement(self, _timeout_seconds: float) -> bool:
        return True

    def sample_now(self) -> int:
        self.samples.append(50_000)
        return 50_000

    def phase_metric_values(self, _started: float) -> dict:
        return {
            "host_temperature_start_millicelsius": 50_000,
            "host_temperature_end_millicelsius": 50_000,
            "host_temperature_peak_millicelsius": 50_000,
            "host_temperature_sample_count": 4,
            "host_thermal_guard_trip_count": 0,
            "host_thermal_pacing_event_count": 0,
            "host_thermal_pacing_completed_event_count": 0,
            "host_thermal_pacing_seconds": 0.0,
        }

    def metric_values(self) -> dict:
        return {
            "host_temperature_end_millicelsius": 50_000,
            "host_temperature_peak_millicelsius": 50_000,
            "host_temperature_start_millicelsius": 50_000,
            "host_thermal_guard_trip_count": 0,
            "host_thermal_cooldown_active_end": 0,
            "host_thermal_cooldown_completed_count": 1,
            "host_thermal_cooldown_peak_millicelsius": 50_000,
            "host_thermal_cooldown_sample_count": 2,
            "host_thermal_cooldown_seconds": 0.25,
            "host_thermal_cooldown_stable_sample_count": 2,
            "host_thermal_cooldown_timeout_count": 0,
        }

    def pacing_metric_values(self) -> dict:
        return {
            "host_thermal_pacing_active_end": 0,
            "host_thermal_pacing_completed_event_count": 0,
            "host_thermal_pacing_event_count": 0,
            "host_thermal_pacing_max_seconds": 0.0,
            "host_thermal_pacing_max_start_millicelsius": 0,
            "host_thermal_pacing_seconds": 0.0,
        }


class FakeStartupTrippedThermalGuard(FakeThermalGuard):
    def start(self) -> None:
        self.trip_reason = "injected startup thermal trip"

    def wait_for_pacing_settlement(self, _timeout_seconds: float) -> bool:
        return False

    def metric_values(self) -> dict:
        values = super().metric_values()
        values["host_thermal_guard_trip_count"] = 1
        return values


class FakeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.bodies: list[dict] = []
        self.counters = {field: 0 for field in bench.COUNTER_FIELDS}
        self.execution_identity = {
            "provenance_type": "kiln.execution-provenance.v1",
            "backend": "test",
            "device": "test:0",
            "inference_dtype": "f32",
            "training_policy": "test",
            "provenance_sha256": "sha256:" + "1" * 64,
            "executable_sha256": "sha256:" + "2" * 64,
            "numerical_runtime_sha256": "sha256:" + "3" * 64,
            "kernel_contract_sha256": "sha256:" + "4" * 64,
            "effective_server_config_sha256": "sha256:" + "5" * 64,
            "effective_environment_sha256": "sha256:" + "6" * 64,
        }

    def health(self) -> dict:
        with self.lock:
            snapshot = {
                "max_decode_batch": 8,
                "max_observed_batch_size": self.max_active,
                **self.counters,
            }
        return {
            "version": "test-v1",
            "execution_identity": self.execution_identity,
            "decode_runtime": {"batching_engine": snapshot},
        }


class FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    state: FakeState

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if self.path == "/v1/models":
            value = {"object": "list", "data": [{"id": "test-model"}]}
        elif self.path == "/health":
            value = self.state.health()
        else:
            self.send_error(404)
            return
        payload = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        max_tokens = body["max_tokens"]
        with self.state.lock:
            self.state.bodies.append(body)
            self.state.active += 1
            self.state.max_active = max(self.state.max_active, self.state.active)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self._event(
                {
                    "model": "test-model",
                    "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}],
                }
            )
            for token in range(max_tokens):
                time.sleep(0.004)
                self._event(
                    {
                        "model": "test-model",
                        "choices": [
                            {
                                "delta": {"content": f"token-{token} "},
                                "finish_reason": "length" if token + 1 == max_tokens else None,
                            }
                        ],
                    }
                )
            self._event(
                {
                    "model": "test-model",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 42,
                        "completion_tokens": max_tokens,
                        "total_tokens": 42 + max_tokens,
                    },
                }
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        finally:
            with self.state.lock:
                self.state.active -= 1
                self.state.counters["total_decode_forwards"] += max_tokens
                self.state.counters["total_batched_decode_forwards"] += max_tokens
                self.state.counters["total_decode_rows"] += max_tokens
                self.state.counters["total_decode_tokens"] += max_tokens
                self.state.counters["total_decode_forward_ms"] += max_tokens * 0.25
                self.state.counters["total_prefill_forwards"] += 1
                self.state.counters["total_prefill_tokens"] += 42
                self.state.counters["total_prefill_layers"] += 8
                self.state.counters["total_prefill_layer_yields"] += 1
                self.state.counters["total_prefill_forward_ms"] += 0.5
                self.state.counters["total_admission_calls"] += 1
                self.state.counters["total_admission_ms"] += 0.1

    def _event(self, value: dict) -> None:
        self.wfile.write(f"data: {json.dumps(value)}\n\n".encode())
        self.wfile.flush()


class FakeServer:
    def __enter__(self) -> "FakeServer":
        state = FakeState()
        handler = type("BoundFakeHandler", (FakeHandler,), {"state": state})
        self.state = state
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"


class ServingBenchmarkTests(unittest.TestCase):
    def _run_cli_fixture(
        self,
        fake: FakeServer,
        directory: str,
        *,
        fetch_json: object | None = None,
        guarded: bool = True,
        thermal_guard_factory: object = FakeThermalGuard,
    ) -> tuple[int, Path]:
        output = Path(directory) / "receipt.json"
        runtime_artifact = Path(directory) / "kiln-server"
        runtime_artifact.write_bytes(b"test runtime")
        fake.state.execution_identity["executable_sha256"] = (
            "sha256:" + hashlib.sha256(b"test runtime").hexdigest()
        )
        memory_counter = Path(directory) / "vram-used"
        memory_counter.write_text("1024")
        thermal_policy = Path(directory) / "host-thermal-policy.json"
        thermal_policy.write_text(json.dumps(valid_host_thermal_policy()))
        model_path = Path(directory) / "model"
        model_fingerprint = {
            "id": "test-model",
            "path": str(model_path),
            "weight_files": [
                {
                    "path": "model.safetensors",
                    "bytes": 8,
                    "sha256": "sha256:" + "c" * 64,
                }
            ],
            "config_hash": "sha256:" + "d" * 64,
            "tokenizer_hash": "sha256:" + "e" * 64,
            "chat_template_hash": "sha256:" + "f" * 64,
        }
        clean_repository = {
            "commit": "a" * 40,
            "dirty": False,
            "source_tree_sha256": "sha256:" + "b" * 64,
        }
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    bench, "repository_identity", return_value=clean_repository
                )
            )
            stack.enter_context(
                mock.patch.object(
                    bench, "fingerprint_model", return_value=model_fingerprint
                )
            )
            if fetch_json is not None:
                stack.enter_context(
                    mock.patch.object(bench, "fetch_json", side_effect=fetch_json)
                )
            stack.enter_context(
                mock.patch.object(
                    bench.AttachedProcessGroup,
                    "attach",
                    return_value=FakeAttachedProcessGroup(),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    bench.thermal,
                    "HostThermalGuard",
                    side_effect=thermal_guard_factory,
                )
            )
            thermal_args = (
                [
                    "--host-thermal-policy",
                    str(thermal_policy),
                    "--server-pid",
                    "4321",
                ]
                if guarded
                else ["--unsafe-no-host-thermal-guard"]
            )
            return_code = bench.main(
                [
                    "--base-url",
                    fake.base_url,
                    "--model",
                    "test-model",
                    "--runtime-identity",
                    "test-runtime",
                    "--runtime-artifact",
                    str(runtime_artifact),
                    "--model-path",
                    str(model_path),
                    "--run-id",
                    "cli-fixture-v1",
                    "--sizes",
                    "1",
                    "--max-tokens",
                    "3",
                    "--warmup-requests",
                    "0",
                    "--memory-path",
                    str(memory_counter),
                    "--memory-limit-bytes",
                    "2048",
                    "--out",
                    str(output),
                    *thermal_args,
                ]
            )
        return return_code, output

    def test_prompts_are_unique_per_row_and_preserve_the_marker_multiset(self) -> None:
        prompts = [
            bench.deterministic_prompt("shared", "measure-c008-r000", i)
            for i in range(8)
        ]
        self.assertEqual(len(prompts), len(set(prompts)))
        marker_rows = [
            prompt.split("Marker sequence: ", 1)[1].removesuffix(".")
            for prompt in prompts
        ]
        expected = sorted(bench.PROMPT_MARKERS)
        for row in marker_rows:
            self.assertEqual(sorted(row.split(" | ")), expected)

    def test_fixed_workload_profiles_pin_sampling_and_prompt_shapes(self) -> None:
        expected = {
            "greedy-short": (0.0, True, "exact_output"),
            "api-default-sampled": (1.0, True, "inputs_only"),
            "long-prefill": (0.0, True, "exact_output"),
            "prefix-hit": (0.0, True, "exact_output"),
            "mixed": (0.0, False, "exact_output"),
        }
        for profile, (temperature, uniform, comparison_mode) in expected.items():
            args = bench.parse_args(["--workload-profile", profile])
            self.assertEqual(args.temperature, temperature)
            self.assertEqual(args.top_p, 1.0)
            self.assertEqual(args.require_uniform_prompt_tokens, uniform)
            self.assertEqual(
                bench.PROFILE_CONTRACTS[profile]["comparison_mode"], comparison_mode
            )

        short = bench.deterministic_prompt("shared", "phase", 0, "short")
        long_prompt = bench.deterministic_prompt(
            "shared", "phase", 0, "long-prefill"
        )
        prefix_prompts = [
            bench.deterministic_prompt("shared", "phase", index, "prefix-hit")
            for index in range(2)
        ]
        mixed_lengths = {
            len(bench.deterministic_prompt("shared", "phase", index, "mixed"))
            for index in range(4)
        }
        self.assertGreater(len(long_prompt), len(short) * 10)
        self.assertNotEqual(prefix_prompts[0], prefix_prompts[1])
        shared_prefix = "Shared prefix for a cache-reuse workload. " + (
            bench.LONG_PROMPT_BLOCK * bench.LONG_PROMPT_REPETITIONS
        )
        self.assertTrue(all(prompt.startswith(shared_prefix) for prompt in prefix_prompts))
        self.assertEqual(len(mixed_lengths), 4)

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(
                ["--workload-profile", "api-default-sampled", "--temperature", "0"]
            )

    def test_sse_parser_handles_comments_and_multiline_data(self) -> None:
        parser = bench.SSEParser()
        self.assertEqual(parser.feed_line(": heartbeat\n"), [])
        self.assertEqual(parser.feed_line("data: first\n"), [])
        self.assertEqual(parser.feed_line("data: second\r\n"), [])
        self.assertEqual(parser.feed_line("\n"), ["first\nsecond"])
        self.assertEqual(parser.finish(), [])

    def test_percentile_uses_r7_interpolation(self) -> None:
        self.assertEqual(bench.percentile_r7([], 0.99), None)
        self.assertEqual(bench.percentile_r7([5.0], 0.99), 5.0)
        self.assertAlmostEqual(bench.percentile_r7([0.0, 10.0], 0.9), 9.0)

    def test_concurrent_stream_run_is_fail_closed_and_captures_diagnostics(self) -> None:
        with FakeServer() as fake:
            args = bench.parse_args(
                [
                    "--base-url",
                    fake.base_url,
                    "--model",
                    "test-model",
                    "--runtime-identity",
                    "test-runtime",
                    "--run-id",
                    "fixture-v1",
                    "--sizes",
                    "4",
                    "--max-tokens",
                    "3",
                    "--warmup-requests",
                    "0",
                    "--max-dispatch-spread-ms",
                    "1000",
                ]
            )
            sampler = bench.MemorySampler(None, 50)
            result = bench.run_once(
                args=args,
                concurrency=4,
                repeat=0,
                max_tokens=3,
                phase="measure-c004-r000",
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                sampler=sampler,
                diagnostics_url=f"{fake.base_url}/health",
            )

        self.assertEqual(result["verdict"], "passed", result)
        self.assertEqual(result["success_count"], 4)
        self.assertEqual(result["completion_tokens"], 12)
        self.assertEqual(result["prompt_tokens_min"], 42)
        self.assertEqual(result["prompt_tokens_max"], 42)
        self.assertGreater(result["output_token_throughput_per_s"], 0)
        self.assertIsNotNone(result["ttft_ms_p99"])
        self.assertIsNotNone(result["client_visible_itl_ms_p99"])
        self.assertEqual(result["server"]["effective_max_decode_batch"], 8)
        self.assertGreaterEqual(result["server"]["process_max_observed_batch"], 2)
        self.assertEqual(result["server"]["total_errors"], 0)
        self.assertEqual(len(fake.state.bodies), 4)
        self.assertEqual(
            len({body["messages"][0]["content"] for body in fake.state.bodies}),
            4,
        )
        for body in fake.state.bodies:
            self.assertEqual(body["model"], "test-model")
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(body["presence_penalty"], 0.0)
            self.assertEqual(body["frequency_penalty"], 0.0)
            self.assertEqual(body["repetition_penalty"], 1.0)

    def test_zero_token_success_is_rejected(self) -> None:
        now = time.perf_counter()
        result = bench.RequestResult(
            index=0,
            prompt_sha256="sha256:prompt",
            started=now,
            ended=now + 0.1,
            semantic_times=[now + 0.05],
            content="text",
            reasoning_content="",
            prompt_tokens=10,
            completion_tokens=0,
            total_tokens=10,
            finish_reason="length",
            done=True,
            error=None,
        )
        summary = bench.summarize_run(
            concurrency=1,
            repeat=0,
            elapsed_s=0.1,
            results=[result],
            max_tokens=1,
            require_max_tokens=True,
            require_uniform_prompt_tokens=True,
            require_nonuniform_prompt_tokens=False,
            max_dispatch_spread_ms=1.0,
            slo_ttft_ms=1000.0,
            slo_itl_ms=1000.0,
            slo_e2e_ms=1000.0,
            memory=None,
            require_memory=False,
            memory_limit_bytes=None,
            server=None,
            diagnostics_error=None,
        )
        self.assertEqual(summary["verdict"], "failed")
        checks = {row["name"]: row["passed"] for row in summary["gates"]}
        self.assertFalse(checks["positive_completion_usage"])
        self.assertFalse(checks["fixed_output_length"])

    def test_absolute_memory_limit_is_a_verdict_gate(self) -> None:
        now = time.perf_counter()
        result = bench.RequestResult(
            index=0,
            prompt_sha256="sha256:prompt",
            started=now,
            ended=now + 0.1,
            semantic_times=[now + 0.05],
            content="text",
            reasoning_content="",
            prompt_tokens=10,
            completion_tokens=1,
            total_tokens=11,
            finish_reason="length",
            done=True,
            error=None,
        )
        summary = bench.summarize_run(
            concurrency=1,
            repeat=0,
            elapsed_s=0.1,
            results=[result],
            max_tokens=1,
            require_max_tokens=True,
            require_uniform_prompt_tokens=True,
            require_nonuniform_prompt_tokens=False,
            max_dispatch_spread_ms=1.0,
            slo_ttft_ms=1000.0,
            slo_itl_ms=1000.0,
            slo_e2e_ms=1000.0,
            memory={
                "baseline_bytes": 100,
                "peak_bytes": 201,
                "peak_delta_bytes": 101,
                "samples": 2,
            },
            require_memory=True,
            memory_limit_bytes=200,
            server=None,
            diagnostics_error=None,
        )
        checks = {row["name"]: row["passed"] for row in summary["gates"]}
        self.assertEqual(summary["verdict"], "failed")
        self.assertFalse(checks["absolute_memory_limit"])

    def test_failed_run_remains_a_valid_structured_counterexample(self) -> None:
        now = time.perf_counter()
        result = bench.failed_result(
            0,
            "sha256:prompt",
            now,
            RuntimeError("injected request failure"),
        )
        summary = bench.summarize_run(
            concurrency=1,
            repeat=0,
            elapsed_s=0.1,
            results=[result],
            max_tokens=1,
            require_max_tokens=True,
            require_uniform_prompt_tokens=True,
            require_nonuniform_prompt_tokens=False,
            max_dispatch_spread_ms=1.0,
            slo_ttft_ms=1000.0,
            slo_itl_ms=1000.0,
            slo_e2e_ms=1000.0,
            memory={
                "baseline_bytes": 100,
                "peak_bytes": 100,
                "peak_delta_bytes": 0,
                "samples": 2,
            },
            require_memory=True,
            memory_limit_bytes=200,
            server=None,
            diagnostics_error=None,
        )
        self.assertEqual(summary["verdict"], "failed")
        self.assertEqual(
            summary["errors"],
            [{"index": 0, "error": "RuntimeError: injected request failure"}],
        )
        summary["host_thermal"] = None
        bench.validate_benchmark_run(
            summary,
            label="counterexample",
            concurrency=1,
            repeat=0,
            max_tokens=1,
            driver_version=bench.DRIVER_VERSION,
            memory_limit_bytes=200,
            workload_profile="greedy-short",
        )

    def test_batching_counter_regression_is_rejected(self) -> None:
        before = {field: 1 for field in bench.COUNTER_FIELDS}
        after = {field: 1 for field in bench.COUNTER_FIELDS}
        before.update(max_decode_batch=8, max_observed_batch_size=4)
        after.update(max_decode_batch=8, max_observed_batch_size=4)
        after["total_decode_rows"] = 0
        with self.assertRaisesRegex(bench.BenchmarkError, "regressed"):
            bench.batching_delta(before, after)

    def test_reference_comparison_binds_workload_prompts_and_outputs(self) -> None:
        current = {
            "schema": bench.SCHEMA,
            "driver_version": bench.DRIVER_VERSION,
            "workload_fingerprint": "sha256:workload",
            "workload": {"comparison_mode": "exact_output"},
            "engine": {"model_identity": {"content_sha256": "sha256:model"}},
            "host_thermal": {
                "policy": {"content_sha256": "sha256:thermal-policy"}
            },
            "runs": [
                {
                    "concurrency": 1,
                    "repeat": 0,
                    "prompt_token_counts": [42],
                    "prompt_set_sha256": "sha256:prompts",
                    "output_set_sha256": "sha256:outputs",
                }
            ],
        }
        reference = json.loads(json.dumps(current))
        reference["engine"]["name"] = "kiln"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(reference))
            with mock.patch.object(bench, "validate_benchmark_receipt"):
                comparison = bench.compare_reference(current, path)
                self.assertTrue(comparison["matched"])
                reference["runs"][0]["output_set_sha256"] = "sha256:different"
                path.write_text(json.dumps(reference))
                comparison = bench.compare_reference(current, path)
                self.assertFalse(comparison["matched"])
                self.assertEqual(
                    comparison["mismatches"][0]["reason"], "output_mismatch"
                )

                reference["workload"]["comparison_mode"] = "inputs_only"
                current["workload"]["comparison_mode"] = "inputs_only"
                path.write_text(json.dumps(reference))
                comparison = bench.compare_reference(current, path)
                self.assertTrue(comparison["matched"])
                reference["runs"][0]["prompt_token_counts"] = [43]
                path.write_text(json.dumps(reference))
                comparison = bench.compare_reference(current, path)
                self.assertEqual(
                    comparison["mismatches"][0]["reason"],
                    "prompt_token_mismatch",
                )

                reference["host_thermal"]["policy"]["content_sha256"] = (
                    "sha256:different-thermal-policy"
                )
                path.write_text(json.dumps(reference))
                with self.assertRaisesRegex(
                    bench.BenchmarkError, "different host thermal policy"
                ):
                    bench.compare_reference(current, path)

                path.write_text('{"schema":"first","schema":"second"}')
                with self.assertRaisesRegex(
                    bench.BenchmarkError, "duplicate JSON object key"
                ):
                    bench.compare_reference(current, path)

    def test_memory_sampler_records_peak_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "used"
            path.write_text("100")
            sampler = bench.MemorySampler(path, 5)
            sampler.start()
            path.write_text("175")
            time.sleep(0.03)
            snapshot = sampler.snapshot()
            sampler.stop()
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["baseline_bytes"], 100)
        self.assertEqual(snapshot["peak_bytes"], 175)
        self.assertEqual(snapshot["peak_delta_bytes"], 75)
        self.assertGreaterEqual(snapshot["samples"], 2)

    def test_host_thermal_policy_is_closed_hashed_and_hysteresis_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            value = valid_host_thermal_policy()
            path.write_text(json.dumps(value))
            record, policy, settlement_timeout = bench.load_host_thermal_policy(path)

            self.assertEqual(record["content_sha256"], bench.canonical_sha256(value))
            self.assertEqual(policy.cooldown_mode, "live_process_safe_handoff")
            self.assertEqual(policy.pacing_resume_stable_samples, 2)
            self.assertEqual(settlement_timeout, 30.0)
            bench.validate_host_thermal_policy_value(record, "fixture")

            legacy_record = json.loads(json.dumps(record))
            legacy_record["pacing"].pop("resume_stable_samples")
            legacy_record.pop("content_sha256")
            legacy_record["content_sha256"] = bench.canonical_sha256(legacy_record)
            _legacy, legacy_policy, _timeout = (
                bench.validate_host_thermal_policy_value(legacy_record, "legacy")
            )
            self.assertEqual(legacy_policy.pacing_resume_stable_samples, 1)

            legacy_input = json.loads(json.dumps(value))
            legacy_input["pacing"].pop("resume_stable_samples")
            with self.assertRaisesRegex(bench.BenchmarkError, "missing keys"):
                bench.validate_host_thermal_policy_value(legacy_input, "input")

            tampered = json.loads(json.dumps(record))
            tampered["pacing"]["start_millicelsius"] = 79_000
            with self.assertRaisesRegex(bench.BenchmarkError, "content_sha256"):
                bench.validate_host_thermal_policy_value(tampered, "fixture")

            value["pacing"]["resume_millicelsius"] = 80_000
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(bench.BenchmarkError, "resume < start"):
                bench.load_host_thermal_policy(path)

    def test_attached_process_group_binds_proc_identity_and_detects_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = root / "4321"
            process.mkdir()
            boot = root / "sys/kernel/random"
            boot.mkdir(parents=True)
            (boot / "boot_id").write_text("fixture-boot-id\n")
            (process / "cmdline").write_bytes(b"kiln\0serve\0")
            (process / "exe").symlink_to("/fixture/kiln-server")

            def write_stat(process_group: int, start_time: int) -> None:
                fields = ["S", "1", str(process_group), *(["0"] * 16), str(start_time)]
                (process / "stat").write_text(
                    f"4321 (server with spaces) {' '.join(fields)}\n"
                )

            write_stat(4321, 123456)
            attached = bench.AttachedProcessGroup.attach(4321, proc_root=root)
            self.assertIsNone(attached.poll())
            self.assertEqual(attached.receipt_identity()["process_group_id"], 4321)

            write_stat(4321, 999999)
            self.assertEqual(attached.poll(), 0)
            write_stat(4000, 123456)
            with self.assertRaisesRegex(bench.BenchmarkError, "lead its process group"):
                bench.AttachedProcessGroup.attach(4321, proc_root=root)

    def test_host_guard_starts_before_the_first_server_probe(self) -> None:
        events: list[str] = []

        class OrderedGuard(FakeThermalGuard):
            def start(self) -> None:
                events.append("guard_started")

        original_fetch_json = bench.fetch_json

        def ordered_fetch_json(*args: object, **kwargs: object) -> dict:
            events.append("server_probe")
            return original_fetch_json(*args, **kwargs)

        with FakeServer() as fake, tempfile.TemporaryDirectory() as directory:
            return_code, _output = self._run_cli_fixture(
                fake,
                directory,
                fetch_json=ordered_fetch_json,
                thermal_guard_factory=OrderedGuard,
            )

        self.assertEqual(return_code, 0)
        self.assertEqual(events[0], "guard_started")
        self.assertIn("server_probe", events[1:])

    def test_owned_server_launch_binds_group_shutdown_and_log_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                reservation.listen()
                port = reservation.getsockname()[1]
                with self.assertRaisesRegex(bench.BenchmarkError, "already listening"):
                    bench.require_owned_base_url_unbound(
                        f"http://127.0.0.1:{port}"
                    )
            executable = root / "fixture-server.py"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import signal\n"
                "import socket\n"
                "import sys\n"
                "import time\n"
                "signal.signal(signal.SIGTERM, lambda *_: exit(0))\n"
                "listener = socket.create_server(('127.0.0.1', int(sys.argv[1])))\n"
                "print('ready', flush=True)\n"
                "while True:\n"
                "    time.sleep(0.1)\n"
            )
            executable.chmod(0o755)
            launch_value = {
                "schema": bench.SERVER_LAUNCH_SCHEMA,
                "id": "fixture-owned-server-v1",
                "command": ["./fixture-server.py", str(port)],
                "working_directory": ".",
                "log_directory": "logs",
                "readiness_poll_interval_ms": 10,
                "startup_timeout_seconds": 5.0,
                "shutdown_timeout_seconds": 5.0,
                "acceptable_exit_codes": [0],
            }
            config = bench.validate_server_launch_config_value(
                launch_value,
                config_directory=root,
                label="fixture",
            )
            base_url = f"http://127.0.0.1:{port}"
            self.assertEqual(bench.require_owned_base_url_unbound(base_url), port)
            original_attach = bench.AttachedProcessGroup.attach
            attach_attempts = 0

            def flaky_attach(pid: int) -> object:
                nonlocal attach_attempts
                attach_attempts += 1
                if attach_attempts == 1:
                    raise bench.BenchmarkError("injected pre-exec identity race")
                return original_attach(pid)

            with mock.patch.object(
                bench.AttachedProcessGroup, "attach", side_effect=flaky_attach
            ):
                server = bench.launch_owned_server(config, "fixture-run-v1")
            try:
                deadline = time.monotonic() + 5.0
                while "ready" not in server.log_path.read_text():
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                bench.verify_owned_listener(server, base_url)
                shutdown = bench.shutdown_owned_server(server)
                log = bench.close_owned_server_log(server)
            finally:
                if server.process.poll() is None:
                    bench.shutdown_owned_server(server)
                if not server.log_handle.closed:
                    server.log_handle.close()

            self.assertEqual(server.identity.pid, server.identity.process_group_id)
            self.assertGreaterEqual(attach_attempts, 2)
            self.assertFalse(shutdown["forced"])
            self.assertEqual(shutdown["returncode"], 0)
            self.assertFalse(shutdown["process_group_alive_end"])
            self.assertGreater(log["bytes"], 0)
            mode, passed = bench.validate_server_lifecycle(
                {
                    "mode": "owned_process_group",
                    "launch_config": config.record,
                    "log": log,
                    "shutdown": shutdown,
                }
            )
            self.assertEqual(mode, "owned_process_group")
            self.assertTrue(passed)

    def test_cli_writes_a_self_hashing_passed_receipt(self) -> None:
        with FakeServer() as fake, tempfile.TemporaryDirectory() as directory:
            return_code, output = self._run_cli_fixture(fake, directory)
            receipt = bench.strict_json_loads(output.read_bytes())
            self.assertEqual(bench.main(["--validate-receipt", str(output)]), 0)

            tampered = json.loads(json.dumps(receipt))
            tampered["unexpected"] = True
            tampered.pop("receipt_sha256")
            tampered["receipt_sha256"] = bench.canonical_sha256(tampered)
            with self.assertRaisesRegex(bench.BenchmarkError, "unknown keys"):
                bench.validate_benchmark_receipt(tampered)

            tampered = json.loads(json.dumps(receipt))
            tampered["engine"]["model_identity"]["config_hash"] = (
                "sha256:" + "0" * 64
            )
            tampered.pop("receipt_sha256")
            tampered["receipt_sha256"] = bench.canonical_sha256(tampered)
            with self.assertRaisesRegex(bench.BenchmarkError, "model content"):
                bench.validate_benchmark_receipt(tampered)

            vllm_receipt = json.loads(json.dumps(receipt))
            vllm_receipt["engine"]["name"] = "vllm"
            vllm_receipt["engine"]["runtime_execution_identity"] = None
            vllm_receipt["engine"]["runtime_manifest"] = valid_vllm_manifest()
            vllm_receipt["completion"]["finalization_checks"][
                "runtime_manifest_unchanged"
            ] = "passed"
            vllm_receipt["completion"]["finalization_checks"][
                "execution_identity_unchanged"
            ] = "not_applicable"
            vllm_receipt.pop("receipt_sha256")
            vllm_receipt["receipt_sha256"] = bench.canonical_sha256(vllm_receipt)
            bench.validate_benchmark_receipt(vllm_receipt)

        self.assertEqual(return_code, 0)
        self.assertEqual(receipt["schema"], bench.SCHEMA)
        self.assertEqual(receipt["verdict"], "passed")
        self.assertEqual(receipt["engine"]["runtime_identity"], "test-runtime")
        self.assertFalse(receipt["engine"]["authentication_configured"])
        self.assertEqual(receipt["engine"]["authentication_source"], "none")
        self.assertEqual(receipt["runs"][0]["success_count"], 1)
        recorded_hash = receipt.pop("receipt_sha256")
        self.assertEqual(recorded_hash, bench.canonical_sha256(receipt))

    def test_cli_preserves_completed_rows_when_final_health_probe_fails(self) -> None:
        with FakeServer() as fake, tempfile.TemporaryDirectory() as directory:
            original_fetch_json = bench.fetch_json
            health_calls = 0

            def fail_final_health(
                url: str, headers: dict[str, str], timeout_secs: float
            ) -> dict:
                nonlocal health_calls
                if url.endswith("/health"):
                    health_calls += 1
                    if health_calls == 4:
                        raise bench.BenchmarkError("injected final health failure")
                return original_fetch_json(url, headers, timeout_secs)

            return_code, output = self._run_cli_fixture(
                fake, directory, fetch_json=fail_final_health
            )
            receipt = bench.strict_json_loads(output.read_bytes())
            bench.validate_benchmark_receipt(receipt)

        self.assertEqual(return_code, 2)
        self.assertEqual(receipt["verdict"], "failed")
        self.assertEqual(len(receipt["runs"]), 1)
        self.assertEqual(receipt["runs"][0]["verdict"], "passed")
        self.assertEqual(receipt["completion"]["expected_run_count"], 1)
        self.assertEqual(receipt["completion"]["completed_run_count"], 1)
        self.assertEqual(
            receipt["completion"]["finalization_checks"][
                "execution_identity_unchanged"
            ],
            "failed",
        )
        self.assertEqual(
            receipt["completion"]["failures"],
            [
                {
                    "phase": "execution_identity_unchanged",
                    "detail": "BenchmarkError: injected final health failure",
                }
            ],
        )

    def test_unsafe_cli_writes_a_valid_failed_diagnostic_receipt(self) -> None:
        with FakeServer() as fake, tempfile.TemporaryDirectory() as directory:
            return_code, output = self._run_cli_fixture(
                fake, directory, guarded=False
            )
            receipt = bench.strict_json_loads(output.read_bytes())
            bench.validate_benchmark_receipt(receipt)

        self.assertEqual(return_code, 2)
        self.assertEqual(receipt["verdict"], "failed")
        self.assertEqual(receipt["runs"][0]["verdict"], "passed")
        self.assertEqual(
            receipt["host_thermal"],
            {
                "mode": "not_configured",
                "unsafe_no_guard_acknowledged": True,
                "policy": None,
                "process_group": None,
                "evidence": None,
            },
        )

    def test_startup_thermal_trip_writes_a_valid_failed_receipt(self) -> None:
        with FakeServer() as fake, tempfile.TemporaryDirectory() as directory:
            return_code, output = self._run_cli_fixture(
                fake,
                directory,
                thermal_guard_factory=FakeStartupTrippedThermalGuard,
            )
            receipt = bench.strict_json_loads(output.read_bytes())
            bench.validate_benchmark_receipt(receipt)

        self.assertEqual(return_code, 2)
        self.assertEqual(receipt["verdict"], "failed")
        self.assertIsNone(receipt["warmup"])
        self.assertEqual(receipt["runs"], [])
        self.assertEqual(
            receipt["completion"]["failures"],
            [
                {
                    "phase": "host_thermal_startup",
                    "detail": "BenchmarkError: injected startup thermal trip",
                },
                {
                    "phase": "host_thermal_handoff",
                    "detail": "BenchmarkError: injected startup thermal trip",
                },
            ],
        )
        self.assertEqual(
            receipt["completion"]["finalization_checks"]["host_thermal_handoff"],
            "failed",
        )

    def test_receipt_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            bench.atomic_write_json(output, {"first": True})
            with self.assertRaisesRegex(bench.BenchmarkError, "refusing to overwrite"):
                bench.atomic_write_json(output, {"second": True})
            self.assertEqual(bench.strict_json_loads(output.read_bytes()), {"first": True})

    def test_sizes_and_nonfinite_cli_values_fail_preflight(self) -> None:
        with self.assertRaisesRegex(bench.BenchmarkError, "strictly increasing"):
            bench.parse_sizes("1,8,8")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(["--temperature", "nan"])

    def test_generic_api_key_is_not_inherited(self) -> None:
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "must-not-leak"}):
            args = bench.parse_args([])
        self.assertIsNone(args.api_key)
        self.assertEqual(args.api_key_source, "none")

        with mock.patch.dict(os.environ, {"BENCH_FIXTURE_KEY": "fixture-secret"}):
            args = bench.parse_args(["--api-key-env", "BENCH_FIXTURE_KEY"])
        self.assertEqual(args.api_key, "fixture-secret")
        self.assertEqual(args.api_key_source, "environment")

    def test_vllm_runtime_manifest_is_canonical_and_model_bound(self) -> None:
        manifest = valid_vllm_manifest()
        self.assertEqual(
            bench.validate_vllm_runtime_manifest(manifest, "fixture"), manifest
        )
        tampered = json.loads(json.dumps(manifest))
        tampered["identity"]["implementation"] = "vllm:changed"
        with self.assertRaisesRegex(bench.BenchmarkError, "canonical_json"):
            bench.validate_vllm_runtime_manifest(tampered, "fixture")

        tampered = json.loads(json.dumps(manifest))
        tampered["system_fingerprint"] = tampered["system_fingerprint"][:-1] + "0"
        with self.assertRaisesRegex(bench.BenchmarkError, "does not bind"):
            bench.validate_vllm_runtime_manifest(tampered, "fixture")

    def test_source_mutation_during_measurement_is_rejected(self) -> None:
        before = {
            "commit": "a" * 40,
            "dirty": False,
            "source_tree_sha256": "sha256:" + "b" * 64,
        }
        after = dict(before, dirty=True)
        with mock.patch.object(bench, "repository_identity", return_value=after):
            with self.assertRaisesRegex(bench.BenchmarkError, "changed during measurement"):
                bench.require_repository_unchanged(before)


if __name__ == "__main__":
    unittest.main()
