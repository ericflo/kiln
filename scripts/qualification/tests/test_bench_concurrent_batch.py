from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
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


class FakeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.bodies: list[dict] = []
        self.counters = {field: 0 for field in bench.COUNTER_FIELDS}

    def health(self) -> dict:
        with self.lock:
            snapshot = {
                "max_decode_batch": 8,
                "max_observed_batch_size": self.max_active,
                **self.counters,
            }
        return {
            "version": "test-v1",
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
    def test_prompts_are_unique_per_row_and_preserve_the_marker_multiset(self) -> None:
        prompts = [bench.deterministic_prompt("shared", "measure-c008-r000", i) for i in range(8)]
        self.assertEqual(len(prompts), len(set(prompts)))
        marker_rows = [prompt.split("Marker sequence: ", 1)[1].removesuffix(".") for prompt in prompts]
        expected = sorted(bench.PROMPT_MARKERS)
        for row in marker_rows:
            self.assertEqual(sorted(row.split(" | ")), expected)

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
            max_dispatch_spread_ms=1.0,
            slo_ttft_ms=1000.0,
            slo_itl_ms=1000.0,
            slo_e2e_ms=1000.0,
            memory=None,
            require_memory=False,
            server=None,
            diagnostics_error=None,
        )
        self.assertEqual(summary["verdict"], "failed")
        checks = {row["name"]: row["passed"] for row in summary["gates"]}
        self.assertFalse(checks["positive_completion_usage"])
        self.assertFalse(checks["fixed_output_length"])

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
            "workload_fingerprint": "sha256:workload",
            "runs": [
                {
                    "concurrency": 1,
                    "repeat": 0,
                    "prompt_set_sha256": "sha256:prompts",
                    "output_set_sha256": "sha256:outputs",
                }
            ],
        }
        reference = json.loads(json.dumps(current))
        reference["engine"] = {"name": "kiln"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(reference))
            comparison = bench.compare_reference(current, path)
            self.assertTrue(comparison["matched"])
            reference["runs"][0]["output_set_sha256"] = "sha256:different"
            path.write_text(json.dumps(reference))
            comparison = bench.compare_reference(current, path)
            self.assertFalse(comparison["matched"])
            self.assertEqual(comparison["mismatches"][0]["reason"], "output_mismatch")

            path.write_text('{"schema":"first","schema":"second"}')
            with self.assertRaisesRegex(bench.BenchmarkError, "duplicate JSON object key"):
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

    def test_cli_writes_a_self_hashing_passed_receipt(self) -> None:
        with FakeServer() as fake, tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            return_code = bench.main(
                [
                    "--base-url",
                    fake.base_url,
                    "--model",
                    "test-model",
                    "--runtime-identity",
                    "test-runtime",
                    "--run-id",
                    "cli-fixture-v1",
                    "--sizes",
                    "1",
                    "--max-tokens",
                    "3",
                    "--warmup-requests",
                    "0",
                    "--memory-path",
                    "none",
                    "--allow-dirty",
                    "--out",
                    str(output),
                ]
            )
            receipt = bench.strict_json_loads(output.read_bytes())

        self.assertEqual(return_code, 0)
        self.assertEqual(receipt["schema"], bench.SCHEMA)
        self.assertEqual(receipt["verdict"], "passed")
        self.assertEqual(receipt["engine"]["runtime_identity"], "test-runtime")
        self.assertFalse(receipt["engine"]["authentication_configured"])
        self.assertEqual(receipt["engine"]["authentication_source"], "none")
        self.assertEqual(receipt["runs"][0]["success_count"], 1)
        recorded_hash = receipt.pop("receipt_sha256")
        self.assertEqual(recorded_hash, bench.canonical_sha256(receipt))

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


if __name__ == "__main__":
    unittest.main()
