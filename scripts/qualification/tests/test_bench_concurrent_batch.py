from __future__ import annotations

import base64
import dataclasses
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

from scripts.qualification.tests.generated_toml import parse_generated_toml


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


def valid_prelaunch_cooldown() -> dict:
    return {
        "scope": "host_package_before_process_creation",
        "sensor_path": "/fixture/temp1_input",
        "poll_interval_ms": 250,
        "target_millicelsius": 65_000,
        "stable_samples_required": 2,
        "stable_samples_observed": 2,
        "timeout_seconds": 30.0,
        "sample_count": 3,
        "temperature_start_millicelsius": 70_000,
        "temperature_peak_millicelsius": 70_000,
        "temperature_end_millicelsius": 50_000,
        "elapsed_seconds": 0.5,
        "completed": True,
    }


def valid_fingerprint_thermal_evidence() -> dict:
    policy_record, policy, settlement = bench.validate_host_thermal_policy_value(
        valid_host_thermal_policy(), "fixture policy"
    )
    return {
        "phase_settlement_timeout_seconds": settlement,
        "policy": policy_record,
        "prelaunch_cooldown": valid_prelaunch_cooldown(),
        "runtime": {
            "host_temperature_end_millicelsius": 50_000,
            "host_temperature_peak_millicelsius": 60_000,
            "host_temperature_start_millicelsius": 50_000,
            "host_thermal_cooldown_active_end": 0,
            "host_thermal_cooldown_completed_count": 1,
            "host_thermal_cooldown_peak_millicelsius": 60_000,
            "host_thermal_cooldown_sample_count": 4,
            "host_thermal_cooldown_seconds": 0.25,
            "host_thermal_cooldown_stable_sample_count": policy.cooldown_stable_samples,
            "host_thermal_cooldown_timeout_count": 0,
            "host_thermal_guard_trip_count": 0,
            "host_thermal_pacing_active_end": 0,
            "host_thermal_pacing_completed_event_count": 1,
            "host_thermal_pacing_event_count": 1,
            "host_thermal_pacing_max_seconds": 0.1,
            "host_thermal_pacing_max_start_millicelsius": (
                policy.pacing_start_millicelsius or 0
            ),
            "host_thermal_pacing_seconds": 0.1,
        },
        "schema": bench.fingerprint_supervisor.SCHEMA,
        "worker_exit_code": 0,
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

    def prepare_for_process_exit(self) -> None:
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
    def __init__(self, request_route: str = "batching_engine") -> None:
        if request_route not in {
            "batching_engine",
            "direct_streaming",
            "direct_streaming_without_rendezvous",
        }:
            raise ValueError(f"unsupported fake request route: {request_route}")
        self.request_route = request_route
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.bodies: list[dict] = []
        self.counters = {field: 0 for field in bench.COUNTER_FIELDS}
        self.request_counters = {
            field: 0 for field in bench.REQUEST_COUNTER_FIELDS[1:]
        }
        self.decode_batcher_counters = {
            field: 0 for field in bench.DECODE_BATCHER_COUNTER_FIELDS
        }
        self.rocm_graph_counters = {
            field: 0 for field in bench.ROCM_GRAPH_COUNTER_FIELDS
        }
        self.rocm_graph_gauges = {
            field: 0 for field in bench.ROCM_GRAPH_GAUGE_FIELDS
        }
        self.rocm_graph_fallbacks = {
            field: 0
            for field in (*bench.ROCM_GRAPH_FALLBACK_COUNTER_FIELDS, "max_duration_micros")
        }
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
            batching_snapshot = {
                "max_decode_batch": 8,
                "max_observed_batch_size": self.max_active,
                **self.counters,
            }
            request_snapshot = {
                "total": sum(self.request_counters.values()),
                **self.request_counters,
                "active": self.active,
                "active_peak": self.max_active,
            }
            direct_snapshot = {
                **self.decode_batcher_counters,
                "runner_calls_per_token": (
                    self.decode_batcher_counters["runner_calls"]
                    / self.decode_batcher_counters["executed_rows"]
                    if self.decode_batcher_counters["executed_rows"]
                    else None
                ),
                "max_runner_calls_per_token": (
                    1 if self.decode_batcher_counters["executed_rows"] else 0
                ),
                "runner_call_budget_per_token": 2,
                "runner_call_budget_exceeded": False,
                "max_observed_batch": (
                    self.max_active
                    if self.decode_batcher_counters["executed_rows"]
                    else 0
                ),
            }
            actor_active = self.request_route == "batching_engine"
            worker_active = self.request_route != "direct_streaming_without_rendezvous"
        return {
            "version": "test-v1",
            "execution_identity": self.execution_identity,
            "requests": request_snapshot,
            "decode_runtime": {
                "batching_configuration": {
                    "mode": {"effective_enabled": actor_active}
                },
                "direct_decode_rendezvous": {
                    "scope": "direct_streaming_greedy_only",
                    "backend_available": True,
                    "actor_active": actor_active,
                    "worker_active": worker_active,
                    "route_available": not actor_active and worker_active,
                },
                "batching_engine": batching_snapshot if actor_active else None,
                "decode_batcher": direct_snapshot if worker_active else None,
                "rocm_graphs": {
                    "requested": True,
                    "capture_requested": True,
                    "enabled": True,
                    "capture_enabled": True,
                    "state": "enabled",
                    "unavailable_reason": None,
                    **self.rocm_graph_counters,
                    **self.rocm_graph_gauges,
                    "fallbacks": self.rocm_graph_fallbacks,
                },
            },
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
                self.state.request_counters["ok"] += 1
                if self.state.request_route == "batching_engine":
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
                elif self.state.request_route == "direct_streaming":
                    decode_tokens = max(0, max_tokens - 1)
                    self.state.decode_batcher_counters["submitted_jobs"] += decode_tokens
                    self.state.decode_batcher_counters["executed_batches"] += decode_tokens
                    self.state.decode_batcher_counters["executed_rows"] += decode_tokens
                    self.state.decode_batcher_counters["runner_calls"] += decode_tokens
                if self.state.rocm_graph_counters["capture_attempts"] == 0:
                    self.state.rocm_graph_counters["capture_attempts"] = 1
                    self.state.rocm_graph_counters["capture_successes"] = 1
                    self.state.rocm_graph_counters["cache_admission_successes"] = 1
                    self.state.rocm_graph_counters["graph_slot_create_count"] = 1
                    self.state.rocm_graph_gauges["captured_graph_count"] = 1
                    self.state.rocm_graph_gauges["graph_slot_count"] = 1
                    self.state.rocm_graph_gauges["idle_graph_slot_count"] = 1
                self.state.rocm_graph_counters["replay_attempts"] += max_tokens
                self.state.rocm_graph_counters["replay_successes"] += max_tokens

    def _event(self, value: dict) -> None:
        self.wfile.write(f"data: {json.dumps(value)}\n\n".encode())
        self.wfile.flush()


class FakeServer:
    def __init__(self, request_route: str = "batching_engine") -> None:
        self.request_route = request_route

    def __enter__(self) -> "FakeServer":
        state = FakeState(self.request_route)
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
    @staticmethod
    def _parse_server_config(path: Path) -> dict:
        source = "\n".join(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        )
        return parse_generated_toml(source)

    def test_rocm_graph_disabled_discriminator_changes_one_typed_field(self) -> None:
        config_root = ROOT / "qualification" / "server-config"
        launch_root = ROOT / "qualification" / "server-launch"
        base_name = "kiln-rocm-strix-halo-serving-comparison-v1"
        diagnostic_name = (
            "kiln-rocm-strix-halo-serving-comparison-graph-disabled-v1"
        )
        no_prefix_name = (
            "kiln-rocm-strix-halo-serving-comparison-graph-disabled-"
            "no-prefix-cache-v1"
        )
        no_batching_name = (
            "kiln-rocm-strix-halo-serving-comparison-graph-disabled-"
            "no-prefix-cache-no-batching-v1"
        )
        prefill_256_name = (
            "kiln-rocm-strix-halo-serving-comparison-graph-disabled-"
            "no-prefix-cache-prefill-256-v1"
        )
        repaired_name = "kiln-rocm-strix-halo-serving-comparison-v2"

        base = self._parse_server_config(config_root / f"{base_name}.toml")
        diagnostic = self._parse_server_config(
            config_root / f"{diagnostic_name}.toml"
        )
        self.assertEqual(base["accelerator"]["rocm_graph_mode"], "profile")
        self.assertEqual(
            diagnostic["accelerator"]["rocm_graph_mode"], "disabled"
        )
        diagnostic["accelerator"]["rocm_graph_mode"] = "profile"
        self.assertEqual(diagnostic, base)

        no_prefix = self._parse_server_config(
            config_root / f"{no_prefix_name}.toml"
        )
        self.assertFalse(no_prefix["prefix_cache"]["enabled"])
        no_prefix["prefix_cache"]["enabled"] = True
        graph_disabled = self._parse_server_config(
            config_root / f"{diagnostic_name}.toml"
        )
        self.assertEqual(no_prefix, graph_disabled)

        no_batching = self._parse_server_config(
            config_root / f"{no_batching_name}.toml"
        )
        self.assertEqual(no_batching["batching"]["mode"], "disabled")
        no_batching["batching"]["mode"] = "enabled"
        no_prefix = self._parse_server_config(
            config_root / f"{no_prefix_name}.toml"
        )
        self.assertEqual(no_batching, no_prefix)

        prefill_256 = self._parse_server_config(
            config_root / f"{prefill_256_name}.toml"
        )
        self.assertEqual(
            prefill_256["server"]["max_prefill_tokens_per_cycle"], 256
        )
        prefill_256["server"]["max_prefill_tokens_per_cycle"] = 64
        no_prefix = self._parse_server_config(
            config_root / f"{no_prefix_name}.toml"
        )
        self.assertEqual(prefill_256, no_prefix)

        repaired = self._parse_server_config(
            config_root / f"{repaired_name}.toml"
        )
        self.assertEqual(
            repaired["server"]["max_prefill_tokens_per_cycle"], 256
        )
        repaired["server"]["max_prefill_tokens_per_cycle"] = 64
        self.assertEqual(repaired, base)

        for name in (
            base_name,
            diagnostic_name,
            no_prefix_name,
            no_batching_name,
            prefill_256_name,
            repaired_name,
        ):
            path = launch_root / f"{name}.json"
            raw = bench.strict_json_loads(path.read_bytes())
            parsed = bench.validate_server_launch_config_value(
                raw,
                config_directory=path.parent,
                label=name,
                require_local_paths=False,
            )
            self.assertEqual(parsed.record["id"], name)
            self.assertEqual(
                parsed.record["command"][-1],
                f"qualification/server-config/{name}.toml",
            )

    def test_rocm_prefill_layer_discriminators_change_one_typed_field(self) -> None:
        config_root = ROOT / "qualification" / "server-config"
        launch_root = ROOT / "qualification" / "server-launch"
        base_name = "kiln-rocm-strix-halo-serving-comparison-v2"
        family_prefix = (
            "kiln-rocm-strix-halo-serving-comparison-prefill-layers-"
        )
        base = self._parse_server_config(config_root / f"{base_name}.toml")
        self.assertEqual(base["server"]["max_prefill_layers_per_cycle"], 4)

        for layer_count in (4, 8, 16, 32):
            launch_name = f"{family_prefix}{layer_count}-v1"
            launch_path = launch_root / f"{launch_name}.json"
            raw = bench.strict_json_loads(launch_path.read_bytes())
            parsed = bench.validate_server_launch_config_value(
                raw,
                config_directory=launch_path.parent,
                label=launch_name,
                require_local_paths=False,
            )
            self.assertEqual(parsed.record["id"], launch_name)
            self.assertEqual(
                parsed.record["log_directory"],
                f"../../.qualification/kiln-serving/logs/"
                f"prefill-layers-{layer_count}-v1",
            )

            if layer_count == 4:
                self.assertEqual(
                    parsed.record["command"][-1],
                    f"qualification/server-config/{base_name}.toml",
                )
                continue

            config_name = f"{family_prefix}{layer_count}-v1"
            self.assertEqual(
                parsed.record["command"][-1],
                f"qualification/server-config/{config_name}.toml",
            )
            candidate = self._parse_server_config(
                config_root / f"{config_name}.toml"
            )
            self.assertEqual(
                candidate["server"]["max_prefill_layers_per_cycle"],
                layer_count,
            )
            candidate["server"]["max_prefill_layers_per_cycle"] = 4
            self.assertEqual(candidate, base)

    def _run_cli_fixture(
        self,
        fake: FakeServer,
        directory: str,
        *,
        fetch_json: object | None = None,
        guarded: bool = True,
        thermal_guard_factory: object = FakeThermalGuard,
        engine: str = "kiln",
        extra_args: list[str] | None = None,
    ) -> tuple[int, Path]:
        output = Path(directory) / "receipt.json"
        runtime_artifact = Path(directory) / "kiln-server"
        if engine == "vllm":
            runtime_artifact.write_text(json.dumps(valid_vllm_manifest()))
        else:
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

        def contained_fingerprint(
            _model_path: Path,
            _model_id: str,
            *,
            policy_path: Path | None,
            phase: str,
        ) -> tuple[dict, dict | None]:
            self.assertIn(
                phase,
                {"model-fingerprint-initial", "model-fingerprint-final"},
            )
            return (
                model_fingerprint,
                valid_fingerprint_thermal_evidence()
                if policy_path is not None
                else None,
            )

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    bench, "repository_identity", return_value=clean_repository
                )
            )
            stack.enter_context(
                mock.patch.object(
                    bench,
                    "fingerprint_model_with_thermal_containment",
                    side_effect=contained_fingerprint,
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
            arguments = [
                    "--engine",
                    engine,
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
            arguments.extend(extra_args or [])
            return_code = bench.main(arguments)
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
        self.assertEqual(result["server"]["request_route"], "batching_engine")
        self.assertEqual(
            result["server"]["batching_engine"]["effective_max_decode_batch"], 8
        )
        self.assertGreaterEqual(
            result["server"]["batching_engine"]["process_max_observed_batch"], 2
        )
        self.assertEqual(result["server"]["batching_engine"]["total_errors"], 0)
        self.assertEqual(result["server"]["requests"]["ok"], 4)
        self.assertEqual(result["server"]["schema"], bench.SERVER_DIAGNOSTICS_SCHEMA)
        self.assertEqual(result["server"]["rocm_graphs"]["capture_successes"], 1)
        self.assertEqual(result["server"]["rocm_graphs"]["replay_successes"], 12)
        self.assertEqual(result["server"]["rocm_graphs"]["fallbacks"]["total"], 0)
        checks = {gate["name"]: gate["passed"] for gate in result["gates"]}
        self.assertTrue(checks["rocm_graph_execution_accounted"])
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

    def test_direct_stream_run_uses_route_aware_diagnostics(self) -> None:
        with FakeServer("direct_streaming") as fake:
            args = bench.parse_args(
                [
                    "--base-url",
                    fake.base_url,
                    "--model",
                    "test-model",
                    "--runtime-identity",
                    "test-runtime",
                    "--run-id",
                    "direct-fixture-v1",
                    "--sizes",
                    "1",
                    "--max-tokens",
                    "3",
                    "--warmup-requests",
                    "0",
                    "--max-dispatch-spread-ms",
                    "1000",
                ]
            )
            result = bench.run_once(
                args=args,
                concurrency=1,
                repeat=0,
                max_tokens=3,
                phase="measure-c001-r000",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                sampler=bench.MemorySampler(None, 50),
                diagnostics_url=f"{fake.base_url}/health",
            )

        self.assertEqual(result["verdict"], "passed", result)
        server = result["server"]
        self.assertEqual(server["request_route"], "direct_streaming")
        self.assertIsNone(server["batching_engine"])
        self.assertTrue(
            server["routing"]["direct_decode_rendezvous"]["route_available"]
        )
        self.assertEqual(server["decode_batcher"]["submitted_jobs"], 2)
        self.assertEqual(server["decode_batcher"]["failed_jobs"], 0)
        self.assertEqual(server["requests"]["ok"], 1)
        checks = {gate["name"]: gate["passed"] for gate in result["gates"]}
        self.assertTrue(checks["server_reported_no_errors"])
        self.assertTrue(checks["server_request_accounting"])

    def test_route_diagnostics_reject_worker_state_without_counters(self) -> None:
        health = FakeState("direct_streaming").health()
        health["decode_runtime"]["decode_batcher"] = None
        with self.assertRaisesRegex(
            bench.BenchmarkError, "decode-batcher diagnostics disagree"
        ):
            bench.server_diagnostics_snapshot(health)

    def test_direct_stream_without_rendezvous_retains_request_diagnostics(self) -> None:
        state = FakeState("direct_streaming_without_rendezvous")
        before = bench.server_diagnostics_snapshot(state.health())
        state.request_counters["ok"] = 1
        after = bench.server_diagnostics_snapshot(state.health())
        server = bench.server_diagnostics_delta(before, after)

        self.assertEqual(server["request_route"], "direct_streaming")
        self.assertIsNone(server["batching_engine"])
        self.assertIsNone(server["decode_batcher"])
        self.assertFalse(
            server["routing"]["direct_decode_rendezvous"]["route_available"]
        )
        self.assertTrue(bench.server_diagnostics_has_no_errors(server))
        self.assertTrue(bench.server_request_accounting_matches(server, 1))
        bench.validate_server_diagnostics_v3(server, "direct fixture")

    def test_driver_v9_server_diagnostics_remain_strict_valid(self) -> None:
        state = FakeState()
        before = bench.server_diagnostics_snapshot(state.health())
        state.request_counters["ok"] = 1
        after = bench.server_diagnostics_snapshot(state.health())
        server = bench.server_diagnostics_delta(before, after)
        legacy = {key: value for key, value in server.items() if key != "rocm_graphs"}
        legacy["schema"] = bench.SERVER_DIAGNOSTICS_SCHEMA_V2

        bench.validate_server_diagnostics_v2(legacy, "driver-v9 fixture")

    def test_rocm_graph_diagnostics_reject_regressed_counters(self) -> None:
        state = FakeState()
        state.rocm_graph_counters["replay_attempts"] = 2
        state.rocm_graph_counters["replay_successes"] = 2
        before = bench.server_diagnostics_snapshot(state.health())
        state.rocm_graph_counters["replay_attempts"] = 1
        state.rocm_graph_counters["replay_successes"] = 1
        after = bench.server_diagnostics_snapshot(state.health())

        with self.assertRaisesRegex(bench.BenchmarkError, "replay_attempts regressed"):
            bench.server_diagnostics_delta(before, after)

    def test_rocm_graph_fallback_fails_execution_accounting_gate(self) -> None:
        state = FakeState()
        before = bench.server_diagnostics_snapshot(state.health())
        state.rocm_graph_counters["capture_attempts"] = 1
        state.rocm_graph_counters["capture_deferrals"] = 1
        state.rocm_graph_fallbacks["total"] = 1
        state.rocm_graph_fallbacks["cold_cache_host_round_trip"] = 1
        after = bench.server_diagnostics_snapshot(state.health())
        server = bench.server_diagnostics_delta(before, after)

        bench.validate_server_diagnostics_v3(server, "fallback fixture")
        self.assertFalse(bench.server_rocm_graph_execution_accounted(server))

    def test_backend_without_rocm_graph_runner_is_explicitly_unavailable(self) -> None:
        state = FakeState()
        health = state.health()
        graph = health["decode_runtime"]["rocm_graphs"]
        graph["state"] = "unavailable"
        graph["unavailable_reason"] = "backend_without_graph_runner"
        for field in (
            "requested",
            "capture_requested",
            "enabled",
            "capture_enabled",
            *bench.ROCM_GRAPH_COUNTER_FIELDS,
            *bench.ROCM_GRAPH_GAUGE_FIELDS,
            "fallbacks",
        ):
            graph[field] = None

        before = bench.server_diagnostics_snapshot(health)
        server = bench.server_diagnostics_delta(before, before)

        bench.validate_server_diagnostics_v3(server, "unavailable fixture")
        self.assertTrue(bench.server_rocm_graph_execution_accounted(server))
        self.assertIsNone(server["rocm_graphs"]["replay_successes"])

        server["rocm_graphs"]["state"] = "busy"
        server["rocm_graphs"]["unavailable_reason"] = "graph_runner_busy"
        bench.validate_server_diagnostics_v3(server, "busy fixture")
        self.assertFalse(bench.server_rocm_graph_execution_accounted(server))

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
                    "output_evidence": [
                        {
                            "index": 0,
                            "output_sha256": "sha256:" + "a" * 64,
                            "reasoning_sha256": "sha256:" + "b" * 64,
                            "content_sha256": "sha256:" + "c" * 64,
                            "reasoning_utf8_bytes": 0,
                            "content_utf8_bytes": 4,
                            "completion_tokens": 1,
                            "finish_reason": "length",
                            "exact_output": None,
                        }
                    ],
                }
            ],
        }
        reference = json.loads(json.dumps(current))
        reference["driver_version"] = "7"
        reference["engine"]["name"] = "kiln"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(reference))
            with mock.patch.object(bench, "validate_benchmark_receipt"):
                comparison = bench.compare_reference(current, path)
                self.assertTrue(comparison["matched"])
                reference["runs"][0]["output_set_sha256"] = "sha256:different"
                reference["runs"][0]["output_evidence"][0]["output_sha256"] = (
                    "sha256:" + "d" * 64
                )
                path.write_text(json.dumps(reference))
                comparison = bench.compare_reference(current, path)
                self.assertFalse(comparison["matched"])
                self.assertEqual(
                    comparison["mismatches"][0]["reason"], "output_mismatch"
                )
                self.assertEqual(comparison["mismatches"][0]["mismatch_count"], 1)
                self.assertEqual(
                    comparison["mismatches"][0]["mismatched_request_indices"], [0]
                )
                self.assertFalse(
                    comparison["mismatches"][0]["request_mismatches"][0][
                        "exact_output_compared"
                    ]
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

    def test_model_fingerprint_runs_in_guarded_closed_worker(self) -> None:
        raw_identity = {
            "id": "test-model",
            "path": "/models/test-model",
            "weight_files": [],
            "config_hash": "sha256:" + "a" * 64,
            "tokenizer_hash": "sha256:" + "b" * 64,
            "chat_template_hash": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.json"
            policy.write_text(json.dumps(valid_host_thermal_policy()))
            with mock.patch.object(
                bench.fingerprint_supervisor,
                "supervise",
                return_value=(
                    0,
                    json.dumps(raw_identity),
                    "",
                    valid_fingerprint_thermal_evidence(),
                ),
            ) as supervise:
                identity, evidence = bench.fingerprint_model_with_thermal_containment(
                    Path("/models/test-model"),
                    "test-model",
                    policy_path=policy,
                    phase="model-fingerprint-initial",
                )
        self.assertEqual(identity, raw_identity)
        self.assertEqual(evidence, valid_fingerprint_thermal_evidence())
        call = supervise.call_args.kwargs
        self.assertEqual(call["worker_phase"], "model-fingerprint-initial")
        self.assertEqual(call["worker_command"][1], str(bench.MODEL_FINGERPRINT_SCRIPT))
        self.assertNotIn("--start-gate", call["worker_command"])
        self.assertEqual(
            set(call["worker_environment"]),
            {"HOME", "LANG", "LC_ALL", "PATH", "PYTHONHASHSEED", "TMPDIR"},
        )
        inherited = "\0".join(
            f"{key}={value}" for key, value in call["worker_environment"].items()
        )
        self.assertNotIn("HF_TOKEN", inherited)
        self.assertNotIn("KILN_", inherited)

    def test_full_output_evidence_is_bounded_validated_and_locates_divergence(
        self,
    ) -> None:
        now = time.perf_counter()
        expected_result = bench.RequestResult(
            index=0,
            prompt_sha256="sha256:prompt",
            started=now,
            ended=now + 0.1,
            semantic_times=[now + 0.05],
            content="caf\u00e9",
            reasoning_content="alpha",
            prompt_tokens=10,
            completion_tokens=1,
            total_tokens=11,
            finish_reason="length",
            done=True,
            error=None,
        )
        actual_result = dataclasses.replace(
            expected_result,
            content="cafe",
            reasoning_content="alphi",
        )
        expected_evidence = bench.output_evidence(expected_result, "full")
        actual_evidence = bench.output_evidence(actual_result, "full")
        expected_row = {
            "concurrency": 1,
            "repeat": 0,
            "output_evidence": [expected_evidence],
        }
        actual_row = {
            "concurrency": 1,
            "repeat": 0,
            "output_evidence": [actual_evidence],
        }
        mismatch = bench.output_mismatch_detail(actual_row, expected_row)
        request = mismatch["request_mismatches"][0]
        self.assertEqual(mismatch["mismatch_count"], 1)
        self.assertTrue(request["exact_output_compared"])
        self.assertEqual(request["reasoning_first_divergent_utf8_byte"], 4)
        self.assertEqual(request["content_first_divergent_utf8_byte"], 3)

        separator_left = dataclasses.replace(
            expected_result, reasoning_content="a", content="b\x1ec"
        )
        separator_right = dataclasses.replace(
            expected_result, reasoning_content="a\x1eb", content="c"
        )
        left_evidence = bench.output_evidence(separator_left, "hashes")
        right_evidence = bench.output_evidence(separator_right, "hashes")
        self.assertEqual(
            left_evidence["output_sha256"], right_evidence["output_sha256"]
        )
        self.assertNotEqual(
            bench.canonical_sha256(bench.output_set_evidence_row(left_evidence)),
            bench.canonical_sha256(bench.output_set_evidence_row(right_evidence)),
        )

        output_rows = [bench.output_set_evidence_row(expected_evidence)]
        bench.validate_comparison_mismatches([mismatch], "fixture comparison")
        bench.validate_output_evidence(
            [expected_evidence],
            label="fixture",
            concurrency=1,
            success_count=1,
            error_indices=set(),
            completion_tokens=1,
            output_set_sha256=bench.canonical_sha256(output_rows),
        )
        tampered = json.loads(json.dumps(expected_evidence))
        tampered["exact_output"]["content_base64"] = base64.b64encode(
            b"fake"
        ).decode()
        with self.assertRaisesRegex(bench.BenchmarkError, "content byte count disagrees"):
            bench.validate_output_evidence(
                [tampered],
                label="fixture",
                concurrency=1,
                success_count=1,
                error_indices=set(),
                completion_tokens=1,
                output_set_sha256=bench.canonical_sha256(output_rows),
            )

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

    def test_prelaunch_cooldown_requires_consecutive_post_provenance_samples(self) -> None:
        policy_record, policy, _timeout = bench.validate_host_thermal_policy_value(
            valid_host_thermal_policy(), "fixture"
        )
        with tempfile.TemporaryDirectory() as directory:
            hwmon = Path(directory) / "hwmon0"
            hwmon.mkdir()
            (hwmon / "name").write_text("fixture\n")
            (hwmon / "temp1_label").write_text("package\n")
            (hwmon / "temp1_input").write_text("50000\n")
            events: list[str] = []

            def trace(event: str, **_fields: object) -> None:
                events.append(event)

            with mock.patch.object(bench.time, "sleep"):
                evidence = bench.wait_for_prelaunch_cooldown(
                    policy,
                    hwmon_root=Path(directory),
                    trace_callback=trace,
                )

        self.assertEqual(evidence["sample_count"], 2)
        self.assertEqual(evidence["stable_samples_observed"], 2)
        self.assertEqual(evidence["temperature_end_millicelsius"], 50_000)
        self.assertEqual(
            events,
            [
                "host_thermal_prelaunch_cooldown_started",
                "host_thermal_prelaunch_cooldown_completed",
            ],
        )
        self.assertTrue(
            bench.validate_prelaunch_cooldown(evidence, policy_record)
        )

    def test_prelaunch_cooldown_times_out_before_process_creation(self) -> None:
        _record, policy, _timeout = bench.validate_host_thermal_policy_value(
            valid_host_thermal_policy(), "fixture"
        )
        with tempfile.TemporaryDirectory() as directory:
            hwmon = Path(directory) / "hwmon0"
            hwmon.mkdir()
            (hwmon / "name").write_text("fixture\n")
            (hwmon / "temp1_label").write_text("package\n")
            (hwmon / "temp1_input").write_text("70000\n")
            with mock.patch.object(
                bench.time, "monotonic", side_effect=[0.0, 1.0, 31.0]
            ):
                with self.assertRaisesRegex(
                    bench.BenchmarkError, "pre-launch cooldown did not reach"
                ):
                    bench.wait_for_prelaunch_cooldown(
                        policy,
                        hwmon_root=Path(directory),
                    )

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
            lifecycle = {
                "mode": "owned_process_group",
                "launch_config": config.record,
                "prelaunch_cooldown": valid_prelaunch_cooldown(),
                "log": log,
                "shutdown": shutdown,
            }
            thermal_record = bench.validate_host_thermal_policy_value(
                valid_host_thermal_policy(), "fixture"
            )[0]
            mode, passed = bench.validate_server_lifecycle(
                lifecycle,
                host_thermal_policy=thermal_record,
            )
            self.assertEqual(mode, "owned_process_group")
            self.assertTrue(passed)
            invalid_lifecycle = json.loads(json.dumps(lifecycle))
            invalid_lifecycle["prelaunch_cooldown"][
                "temperature_end_millicelsius"
            ] = 70_000
            with self.assertRaisesRegex(
                bench.BenchmarkError, "does not prove stable cooldown"
            ):
                bench.validate_server_lifecycle(
                    invalid_lifecycle,
                    host_thermal_policy=thermal_record,
                )

    def test_owned_server_readiness_exit_retains_complete_failed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "failing-server.py"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "import time\n"
                "print('injected startup failure', flush=True)\n"
                "time.sleep(0.25)\n"
                "sys.exit(1)\n"
            )
            executable.chmod(0o755)
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            launch_path = root / "launch.json"
            launch_path.write_text(
                json.dumps(
                    {
                        "schema": bench.SERVER_LAUNCH_SCHEMA,
                        "id": "fixture-failing-server-v1",
                        "command": ["./failing-server.py"],
                        "working_directory": ".",
                        "log_directory": "logs",
                        "readiness_poll_interval_ms": 10,
                        "startup_timeout_seconds": 2.0,
                        "shutdown_timeout_seconds": 1.0,
                        "acceptable_exit_codes": [0],
                    }
                )
            )
            runtime_artifact = root / "vllm-manifest.json"
            runtime_artifact.write_text(json.dumps(valid_vllm_manifest()))
            memory_counter = root / "vram-used"
            memory_counter.write_text("1024")
            thermal_policy = root / "thermal.json"
            thermal_policy.write_text(json.dumps(valid_host_thermal_policy()))
            output = root / "receipt.json"
            model_fingerprint = {
                "id": "test-model",
                "path": str(root / "model"),
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
            with (
                mock.patch.object(
                    bench, "repository_identity", return_value=clean_repository
                ),
                mock.patch.object(
                    bench,
                    "fingerprint_model_with_thermal_containment",
                    return_value=(
                        model_fingerprint,
                        valid_fingerprint_thermal_evidence(),
                    ),
                ),
                mock.patch.object(
                    bench,
                    "wait_for_prelaunch_cooldown",
                    return_value=valid_prelaunch_cooldown(),
                ),
                mock.patch.object(
                    bench.thermal,
                    "HostThermalGuard",
                    side_effect=FakeThermalGuard,
                ),
            ):
                return_code = bench.main(
                    [
                        "--engine=vllm",
                        f"--base-url=http://127.0.0.1:{port}",
                        "--model=test-model",
                        f"--model-path={root / 'model'}",
                        "--runtime-identity=test-vllm-runtime",
                        f"--runtime-artifact={runtime_artifact}",
                        "--run-id=owned-startup-failure-v1",
                        "--sizes=1",
                        "--max-tokens=1",
                        "--warmup-requests=0",
                        f"--memory-path={memory_counter}",
                        "--memory-limit-bytes=2048",
                        f"--host-thermal-policy={thermal_policy}",
                        f"--server-launch-config={launch_path}",
                        f"--out={output}",
                    ]
                )

            receipt = bench.strict_json_loads(output.read_bytes())
            bench.validate_benchmark_receipt(receipt)
            self.assertEqual(return_code, 2)
            self.assertEqual(receipt["verdict"], "failed")
            self.assertEqual(
                [failure["phase"] for failure in receipt["completion"]["failures"]],
                ["server_startup", "server_shutdown"],
            )
            lifecycle = receipt["server_lifecycle"]
            self.assertEqual(lifecycle["shutdown"]["returncode"], 1)
            self.assertFalse(lifecycle["shutdown"]["signal_sent"])
            self.assertFalse(lifecycle["shutdown"]["forced"])
            self.assertFalse(lifecycle["shutdown"]["process_group_alive_end"])
            self.assertGreater(lifecycle["log"]["bytes"], 0)
            self.assertIsNone(receipt["host_thermal"]["evidence"]["trip_reason"])

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

            tampered = json.loads(json.dumps(receipt))
            tampered["engine"]["available_models"] = []
            tampered.pop("receipt_sha256")
            tampered["receipt_sha256"] = bench.canonical_sha256(tampered)
            with self.assertRaisesRegex(bench.BenchmarkError, "absent"):
                bench.validate_benchmark_receipt(tampered)

            tampered = json.loads(json.dumps(receipt))
            tampered["host_thermal"].pop("model_fingerprint")
            tampered.pop("receipt_sha256")
            tampered["receipt_sha256"] = bench.canonical_sha256(tampered)
            with self.assertRaisesRegex(bench.BenchmarkError, "missing keys"):
                bench.validate_benchmark_receipt(tampered)

            tampered = json.loads(json.dumps(receipt))
            tampered["host_thermal"]["model_fingerprint"]["final"] = None
            tampered.pop("receipt_sha256")
            tampered["receipt_sha256"] = bench.canonical_sha256(tampered)
            with self.assertRaisesRegex(
                bench.BenchmarkError, "disagrees with its finalization check"
            ):
                bench.validate_benchmark_receipt(tampered)

            tampered = json.loads(json.dumps(receipt))
            tampered["host_thermal"]["model_fingerprint"]["initial"]["runtime"][
                "host_temperature_peak_millicelsius"
            ] = 90_000
            tampered.pop("receipt_sha256")
            tampered["receipt_sha256"] = bench.canonical_sha256(tampered)
            with self.assertRaisesRegex(bench.BenchmarkError, "did not close safely"):
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
        with (
            FakeServer() as reference_server,
            tempfile.TemporaryDirectory() as reference_directory,
        ):
            reference_code, reference_path = self._run_cli_fixture(
                reference_server, reference_directory
            )
            self.assertEqual(reference_code, 0)

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
                    fake,
                    directory,
                    fetch_json=fail_final_health,
                    extra_args=["--reference-receipt", str(reference_path)],
                )
                receipt = bench.strict_json_loads(output.read_bytes())
                bench.validate_benchmark_receipt(receipt)

        self.assertEqual(return_code, 2)
        self.assertEqual(receipt["verdict"], "failed")
        self.assertEqual(len(receipt["runs"]), 1)
        self.assertEqual(receipt["runs"][0]["verdict"], "passed")
        self.assertEqual(receipt["completion"]["expected_run_count"], 1)
        self.assertEqual(receipt["completion"]["completed_run_count"], 1)
        self.assertTrue(receipt["comparison"]["matched"])
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
                "model_fingerprint": None,
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
                    "phase": "execution_identity_unchanged",
                    "detail": (
                        "BenchmarkError: Kiln execution identity is unavailable "
                        "because startup did not complete"
                    ),
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

    def test_server_startup_failure_writes_a_valid_failed_receipt(self) -> None:
        def fail_models(url: str, *_args: object, **_kwargs: object) -> dict:
            if url.endswith("/v1/models"):
                raise bench.BenchmarkError("injected readiness failure")
            raise AssertionError(f"unexpected fetch after startup failure: {url}")

        for engine in ("kiln", "vllm"):
            with self.subTest(engine=engine):
                with FakeServer() as fake, tempfile.TemporaryDirectory() as directory:
                    return_code, output = self._run_cli_fixture(
                        fake,
                        directory,
                        fetch_json=fail_models,
                        engine=engine,
                    )
                    receipt = bench.strict_json_loads(output.read_bytes())
                    bench.validate_benchmark_receipt(receipt)

                self.assertEqual(return_code, 2)
                self.assertEqual(receipt["verdict"], "failed")
                self.assertEqual(receipt["engine"]["available_models"], [])
                self.assertIsNone(receipt["warmup"])
                self.assertEqual(receipt["runs"], [])
                failure_phases = [
                    failure["phase"]
                    for failure in receipt["completion"]["failures"]
                ]
                self.assertEqual(
                    failure_phases,
                    (
                        ["server_startup", "execution_identity_unchanged"]
                        if engine == "kiln"
                        else ["server_startup"]
                    ),
                )
                self.assertIn(
                    "injected readiness failure",
                    receipt["completion"]["failures"][0]["detail"],
                )
                if engine == "kiln":
                    self.assertIsNone(
                        receipt["engine"]["runtime_execution_identity"]
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
        reordered = json.loads(json.dumps(manifest))
        reordered["identity"] = dict(reversed(tuple(reordered["identity"].items())))
        self.assertEqual(
            bench.validate_vllm_runtime_manifest(reordered, "fixture"), reordered
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
