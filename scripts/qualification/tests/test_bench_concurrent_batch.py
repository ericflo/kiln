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
from scripts import vllm_teacher


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "bench-concurrent-batch.py"
SPEC = importlib.util.spec_from_file_location("bench_concurrent_batch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def valid_vllm_manifest(
    model: str = "test-model",
    *,
    max_model_len: int = 32_768,
) -> dict:
    identity = {
        "schema": "kiln.teacher-identity.v1",
        "served_model_id": model,
        "implementation": "vllm:0.25.0",
        "max_top_k": 20,
        "max_model_len": max_model_len,
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



def valid_request_performance(max_tokens: int, *, batching: bool = True) -> dict:
    gap_samples = max(0, max_tokens - 1)
    gap_stat = 4.0 if gap_samples else None
    phases = {field: None for field in bench.REQUEST_PHASE_FIELDS}
    phases.update(
        actor_queue_ms=0.1,
        actor_admission_ms=0.1,
        prefill_ms=0.5,
        decode_ms=max_tokens * 0.25,
        actor_cycle_idle_ms=max_tokens * 0.5,
        response_delivery_ms=0.1,
        handler_queue_ms=0.1,
        client_delivery_ms=0.1,
        unexplained_ms=0.0,
    )
    reasons = {field: 0 for field in bench.REQUEST_STALL_REASON_FIELDS}
    return {
        "prompt_tokens": 42,
        "completion_tokens": max_tokens,
        "ttft_ms": 4.0,
        "prefill_ms": 0.5,
        "actor_queue_ms": 0.1,
        "actor_admission_ms": 0.1,
        "actor_prefill_wall_ms": 0.5,
        "resident_prefill_used": False if batching else None,
        "decode_ms": max_tokens * 0.25,
        "total_latency_ms": max_tokens * 4.0,
        "decode_tokens_per_sec": 250.0,
        "adapter_used": "none",
        "thinking_mode": "disabled",
        "finish_reason": "length",
        "latency": {
            "emitted_tokens": max_tokens,
            "gap_samples": gap_samples,
            "retained_gap_samples": gap_samples,
            "gap_samples_truncated": False,
            "ttft_ms": 4.0,
            "itl_ms_p50": gap_stat,
            "itl_ms_p99": gap_stat,
            "itl_ms_p999": gap_stat,
            "max_itl_ms": gap_stat,
            "stall_threshold_ms": 20.0 if gap_samples else None,
            "stall_count": 0,
            "unexplained_stall_count": 0,
            "stall_reasons": reasons,
            "phases": phases,
        },
    }


def as_v5_server_diagnostics(server: dict) -> dict:
    legacy = json.loads(json.dumps(server))
    legacy["schema"] = bench.SERVER_DIAGNOSTICS_SCHEMA_V5
    legacy["routing"] = {
        "batching_actor_effective": True,
        "direct_decode_rendezvous": {
            "scope": "retired",
            "backend_available": True,
            "backend_unavailable_reason": None,
            "actor_active": True,
            "worker_active": False,
            "route_available": False,
        },
    }
    legacy["decode_batcher"] = None
    legacy["rocm_graphs"].pop("capture_parity")
    return legacy


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


class FakeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.system_fingerprint: str | None = "fixture-system-fingerprint"
        self.bodies: list[dict] = []
        self.counters = {field: 0 for field in bench.COUNTER_FIELDS}
        self.request_counters = {
            field: 0 for field in bench.REQUEST_COUNTER_FIELDS[1:]
        }
        self.rocm_graph_counters = {
            field: 0 for field in bench.ROCM_GRAPH_COUNTER_FIELDS
        }
        self.rocm_graph_parity_counters = {
            field: 0
            for field in (
                *bench.ROCM_GRAPH_BATCHED_CAPTURE_COUNTER_FIELDS,
                *bench.ROCM_GRAPH_CAPTURE_PARITY_COUNTER_FIELDS,
            )
        }
        self.rocm_graph_gauges = {
            field: 0 for field in bench.ROCM_GRAPH_GAUGE_FIELDS
        }
        self.rocm_graph_fallbacks = {
            field: 0
            for field in (*bench.ROCM_GRAPH_FALLBACK_COUNTER_FIELDS, "max_duration_micros")
        }
        self.actor_cycle_idle_ms = 0
        self.actor_cycle_idle_source = "default"
        self.actor_cycle_idle_active = False
        self.actor_cycle_idle_count = 0
        self.total_actor_cycle_idle_ms = 0.0
        self.max_actor_cycle_idle_ms = 0.0
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
                "actor_cycle_idle_ms": self.actor_cycle_idle_ms,
                "actor_cycle_idle_source": self.actor_cycle_idle_source,
                "actor_cycle_idle_active": self.actor_cycle_idle_active,
                "actor_cycle_idle_count": self.actor_cycle_idle_count,
                "total_actor_cycle_idle_ms": self.total_actor_cycle_idle_ms,
                "max_actor_cycle_idle_ms": self.max_actor_cycle_idle_ms,
                **self.counters,
            }
            request_snapshot = {
                "total": sum(self.request_counters.values()),
                **self.request_counters,
                "active": self.active,
                "active_peak": self.max_active,
            }
        return {
            "version": "test-v1",
            "execution_identity": self.execution_identity,
            "requests": request_snapshot,
            "decode_runtime": {
                "batching_engine": batching_snapshot,
                "rocm_graphs": {
                    "requested": True,
                    "capture_requested": True,
                    "enabled": True,
                    "capture_enabled": True,
                    "state": "enabled",
                    "unavailable_reason": None,
                    **self.rocm_graph_counters,
                    **self.rocm_graph_parity_counters,
                    **self.rocm_graph_gauges,
                    "fallbacks": dict(self.rocm_graph_fallbacks),
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
            first_event = {
                "model": "test-model",
                "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}],
            }
            if self.state.system_fingerprint is not None:
                first_event["system_fingerprint"] = self.state.system_fingerprint
            self._event(first_event)
            for token in range(max_tokens):
                time.sleep(0.004)
                event = {
                    "model": "test-model",
                    "choices": [
                        {
                            "delta": {"content": f"token-{token} "},
                            "finish_reason": "length" if token + 1 == max_tokens else None,
                        }
                    ],
                }
                if token + 1 == max_tokens:
                    event["metadata"] = {
                        "performance": valid_request_performance(
                            max_tokens,
                            batching=True,
                        )
                    }
                self._event(event)
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
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.state.lock:
                self.state.active -= 1
                self.state.request_counters["ok"] += 1
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
                if self.state.rocm_graph_counters["capture_attempts"] == 0:
                    self.state.rocm_graph_counters["capture_attempts"] = 1
                    self.state.rocm_graph_counters["capture_successes"] = 1
                    self.state.rocm_graph_counters["cache_admission_successes"] = 1
                    self.state.rocm_graph_counters["graph_slot_create_count"] = 1
                    self.state.rocm_graph_gauges["captured_graph_count"] = 1
                    self.state.rocm_graph_gauges["graph_slot_count"] = 1
                    self.state.rocm_graph_gauges["idle_graph_slot_count"] = 1
                    self.state.rocm_graph_parity_counters.update(
                        batched_capture_attempts=1,
                        batched_capture_successes=1,
                        capture_parity_checks=1,
                        capture_parity_passes=1,
                        capture_parity_compared_bytes=4096,
                        capture_parity_duration_micros=10,
                    )
                self.state.rocm_graph_counters["replay_attempts"] += max_tokens
                self.state.rocm_graph_counters["replay_successes"] += max_tokens

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

    def test_rocm_decode_width_discriminator_changes_one_typed_field(self) -> None:
        config_root = ROOT / "qualification" / "server-config"
        launch_root = ROOT / "qualification" / "server-launch"
        base_name = "kiln-rocm-strix-halo-serving-comparison-v2"
        candidate_name = (
            "kiln-rocm-strix-halo-serving-comparison-decode-batch-4-v1"
        )
        base = self._parse_server_config(config_root / f"{base_name}.toml")
        candidate = self._parse_server_config(
            config_root / f"{candidate_name}.toml"
        )
        self.assertEqual(base["server"]["max_decode_batch"], 8)
        self.assertEqual(candidate["server"]["max_decode_batch"], 4)
        candidate["server"]["max_decode_batch"] = 8
        self.assertEqual(candidate, base)

        launch_path = launch_root / f"{candidate_name}.json"
        raw = bench.strict_json_loads(launch_path.read_bytes())
        parsed = bench.validate_server_launch_config_value(
            raw,
            config_directory=launch_path.parent,
            label=candidate_name,
            require_local_paths=False,
        )
        self.assertEqual(parsed.record["id"], candidate_name)
        self.assertEqual(
            parsed.record["command"][-1],
            f"qualification/server-config/{candidate_name}.toml",
        )
        self.assertEqual(
            parsed.record["log_directory"],
            "../../.qualification/kiln-serving/logs/decode-batch-4-v1",
        )

    def test_rocm_actor_cycle_idle_discriminator_changes_one_typed_field(self) -> None:
        config_root = ROOT / "qualification" / "server-config"
        launch_root = ROOT / "qualification" / "server-launch"
        parent_name = "kiln-rocm-strix-halo-serving-comparison-decode-batch-4-v1"
        candidate_name = (
            "kiln-rocm-strix-halo-serving-comparison-decode-batch-4-"
            "actor-cycle-idle-100ms-v1"
        )
        parent = self._parse_server_config(config_root / f"{parent_name}.toml")
        candidate = self._parse_server_config(
            config_root / f"{candidate_name}.toml"
        )
        self.assertNotIn("actor_cycle_idle_ms", parent["batching"])
        self.assertEqual(candidate["batching"]["actor_cycle_idle_ms"], 100)
        candidate["batching"].pop("actor_cycle_idle_ms")
        self.assertEqual(candidate, parent)

        launch_path = launch_root / f"{candidate_name}.json"
        parsed = bench.validate_server_launch_config_value(
            bench.strict_json_loads(launch_path.read_bytes()),
            config_directory=launch_path.parent,
            label=candidate_name,
            require_local_paths=False,
        )
        self.assertEqual(parsed.record["id"], candidate_name)
        self.assertEqual(
            parsed.record["command"][-1],
            f"qualification/server-config/{candidate_name}.toml",
        )
        self.assertEqual(
            parsed.record["log_directory"],
            "../../.qualification/kiln-serving/logs/"
            "decode-batch-4-actor-cycle-idle-100ms-v1",
        )

    def test_rocm_prefill_amortization_discriminator_changes_one_typed_field(self) -> None:
        config_root = ROOT / "qualification" / "server-config"
        launch_root = ROOT / "qualification" / "server-launch"
        parent_name = (
            "kiln-rocm-strix-halo-serving-comparison-decode-batch-4-"
            "actor-cycle-idle-100ms-v1"
        )
        candidate_name = (
            "kiln-rocm-strix-halo-serving-comparison-decode-batch-4-"
            "actor-cycle-idle-100ms-prefill-layers-32-v1"
        )
        parent = self._parse_server_config(config_root / f"{parent_name}.toml")
        candidate = self._parse_server_config(
            config_root / f"{candidate_name}.toml"
        )
        self.assertEqual(parent["server"]["max_prefill_layers_per_cycle"], 4)
        self.assertEqual(candidate["server"]["max_prefill_layers_per_cycle"], 32)
        candidate["server"]["max_prefill_layers_per_cycle"] = 4
        self.assertEqual(candidate, parent)

        launch_path = launch_root / f"{candidate_name}.json"
        parsed = bench.validate_server_launch_config_value(
            bench.strict_json_loads(launch_path.read_bytes()),
            config_directory=launch_path.parent,
            label=candidate_name,
            require_local_paths=False,
        )
        self.assertEqual(parsed.record["id"], candidate_name)
        self.assertEqual(
            parsed.record["command"][-1],
            f"qualification/server-config/{candidate_name}.toml",
        )
        self.assertEqual(
            parsed.record["log_directory"],
            "../../.qualification/kiln-serving/logs/"
            "decode-batch-4-actor-cycle-idle-100ms-prefill-layers-32-v1",
        )

    def test_rocm_fixed_source_eager_oracle_changes_only_graph_mode(self) -> None:
        config_root = ROOT / "qualification" / "server-config"
        launch_root = ROOT / "qualification" / "server-launch"
        graph_name = (
            "kiln-rocm-strix-halo-serving-comparison-decode-batch-4-"
            "actor-cycle-idle-100ms-prefill-layers-32-v1"
        )
        eager_name = (
            "kiln-rocm-strix-halo-serving-comparison-decode-batch-4-"
            "actor-cycle-idle-100ms-prefill-layers-32-graph-disabled-v1"
        )
        graph = self._parse_server_config(config_root / f"{graph_name}.toml")
        eager = self._parse_server_config(config_root / f"{eager_name}.toml")
        self.assertEqual(graph["accelerator"]["rocm_graph_mode"], "profile")
        self.assertEqual(eager["accelerator"]["rocm_graph_mode"], "disabled")
        eager["accelerator"]["rocm_graph_mode"] = "profile"
        self.assertEqual(eager, graph)

        launch_path = launch_root / f"{eager_name}.json"
        parsed = bench.validate_server_launch_config_value(
            bench.strict_json_loads(launch_path.read_bytes()),
            config_directory=launch_path.parent,
            label=eager_name,
            require_local_paths=False,
        )
        self.assertEqual(parsed.record["id"], eager_name)
        self.assertEqual(
            parsed.record["command"][-1],
            f"qualification/server-config/{eager_name}.toml",
        )
        self.assertEqual(
            parsed.record["log_directory"],
            "../../.qualification/kiln-serving/logs/"
            "decode-batch-4-actor-cycle-idle-100ms-prefill-layers-32-"
            "graph-disabled-v1",
        )

    def test_cuda_4090_bootstrap_inputs_are_bounded_portable_and_closed(self) -> None:
        config_root = ROOT / "qualification" / "server-config"
        launch_root = ROOT / "qualification" / "server-launch"
        laptop_name = "kiln-cuda-rtx4090-laptop-serving-bootstrap-v1"
        desktop_name = "kiln-cuda-rtx4090-desktop-serving-bootstrap-v1"
        laptop = self._parse_server_config(config_root / f"{laptop_name}.toml")
        desktop = self._parse_server_config(config_root / f"{desktop_name}.toml")

        for name, config, expected in (
            (laptop_name, laptop, (16.0, 1.5, 4_096, 512, 16)),
            (desktop_name, desktop, (23.0, 2.0, 8_192, 1_024, 32)),
        ):
            gpu_gib, floor_gib, batch_tokens, prefill_tokens, decode_batch = expected
            self.assertEqual(config["server"]["serving_profile"], "stable")
            self.assertEqual(config["server"]["max_batch_tokens"], batch_tokens)
            self.assertEqual(
                config["server"]["max_prefill_tokens_per_cycle"], prefill_tokens
            )
            self.assertEqual(config["server"]["max_decode_batch"], decode_batch)
            self.assertEqual(config["accelerator"]["cuda_kernel_profile"], "native_default")
            self.assertEqual(config["accelerator"]["cuda_marlin_profile"], "disabled")
            self.assertEqual(config["accelerator"]["rocm_graph_mode"], "disabled")
            self.assertEqual(config["memory"]["gpu_memory_gb"], gpu_gib)
            self.assertEqual(config["memory"]["floor_gb"], floor_gib)
            self.assertEqual(config["memory"]["reclaim_mode"], "off")
            self.assertFalse(config["memory"]["kv_autoscale"])
            self.assertFalse(config["memory"]["cuda_graphs"])
            self.assertFalse(Path(config["model"]["path"]).is_absolute())
            self.assertFalse(Path(config["model"]["adapter_dir"]).is_absolute())
            self.assertFalse(Path(config["model"]["snapshot_dir"]).is_absolute())

            launch_path = launch_root / f"{name}.json"
            launch = bench.validate_server_launch_config_value(
                bench.strict_json_loads(launch_path.read_bytes()),
                config_directory=launch_path.parent,
                label=name,
                require_local_paths=False,
            )
            self.assertEqual(launch.record["id"], name)
            self.assertEqual(
                launch.record["command"],
                [
                    "./target/release/kiln",
                    "serve",
                    "--config",
                    f"qualification/server-config/{name}.toml",
                ],
            )

        laptop["server"].update(
            max_batch_tokens=8_192,
            max_prefill_tokens_per_cycle=1_024,
            max_decode_batch=32,
        )
        laptop["model"]["adapter_dir"] = desktop["model"]["adapter_dir"]
        laptop["model"]["snapshot_dir"] = desktop["model"]["snapshot_dir"]
        laptop["memory"]["gpu_memory_gb"] = 23.0
        laptop["memory"]["floor_gb"] = 2.0
        self.assertEqual(laptop, desktop)

    def test_metal_macbook_air_bootstrap_inputs_are_bounded_and_closed(self) -> None:
        name = "kiln-metal-macbook-air-m1-serving-bootstrap-v1"
        config = self._parse_server_config(
            ROOT / "qualification" / "server-config" / f"{name}.toml"
        )

        self.assertEqual(config["server"]["serving_profile"], "stable")
        self.assertEqual(config["server"]["port"], 8421)
        self.assertEqual(config["server"]["max_batch_tokens"], 2_048)
        self.assertEqual(config["server"]["max_prefill_tokens_per_cycle"], 512)
        self.assertEqual(config["server"]["max_decode_batch"], 8)
        self.assertEqual(config["accelerator"]["metal_kernel_profile"], "native_default")
        self.assertEqual(config["accelerator"]["rocm_graph_mode"], "disabled")
        self.assertEqual(config["memory"]["gpu_memory_gb"], 12.0)
        self.assertEqual(config["memory"]["floor_gb"], 1.0)
        self.assertEqual(config["memory"]["reclaim_mode"], "off")
        self.assertFalse(config["memory"]["kv_autoscale"])
        self.assertFalse(config["memory"]["cuda_graphs"])
        self.assertEqual(config["model"]["path"], "Qwen3.5-4B")
        self.assertEqual(config["model"]["model_id"], "Qwen/Qwen3.5-4B")
        self.assertEqual(config["model"]["served_model_id"], "Qwen3.5-4B")
        self.assertFalse(Path(config["model"]["path"]).is_absolute())
        self.assertFalse(Path(config["model"]["adapter_dir"]).is_absolute())
        self.assertFalse(Path(config["model"]["snapshot_dir"]).is_absolute())

        launch_path = ROOT / "qualification" / "server-launch" / f"{name}.json"
        launch = bench.validate_server_launch_config_value(
            bench.strict_json_loads(launch_path.read_bytes()),
            config_directory=launch_path.parent,
            label=name,
            require_local_paths=False,
        )
        self.assertEqual(launch.record["id"], name)
        self.assertEqual(
            launch.record["command"],
            [
                "./target/release/kiln",
                "serve",
                "--config",
                f"qualification/server-config/{name}.toml",
            ],
        )

    def test_cuda_vllm_bootstrap_launch_uses_reviewed_immutable_options(self) -> None:
        launch_path = (
            ROOT
            / "qualification"
            / "server-launch"
            / "vllm-cuda-rtx4090-serving-bootstrap-v1.json"
        )
        launch = bench.validate_server_launch_config_value(
            bench.strict_json_loads(launch_path.read_bytes()),
            config_directory=launch_path.parent,
            label=launch_path.stem,
            require_local_paths=False,
        )
        command = launch.record["command"]
        self.assertEqual(command[0], "./.qualification/vllm-cuda-venv/bin/python-kiln")
        self.assertIn("--process-group-mode=inherited", command)
        self.assertIn("--cache-root=.qualification/vllm-runtime-caches", command)
        self.assertNotIn("--attention-backend=TRITON_ATTN", command)
        separator = command.index("--")
        self.assertEqual(
            vllm_teacher.validate_extra_vllm_args(command[separator + 1 :]),
            command[separator + 1 :],
        )
        self.assertIn("--gpu-memory-utilization=0.75", command)
        self.assertIn("--max-num-seqs=64", command)
        self.assertIn("--language-model-only", command)
        args = bench.validate_vllm_owned_launch(
            launch,
            valid_vllm_manifest("Qwen3.5-4B"),
        )
        self.assertEqual(args.served_model_id, "Qwen3.5-4B")
        self.assertEqual(args.process_group_mode, "inherited")

    def test_vllm_owned_launch_rejects_provenance_and_argument_drift(self) -> None:
        launch_path = (
            ROOT
            / "qualification"
            / "server-launch"
            / "vllm-cuda-rtx4090-serving-bootstrap-v1.json"
        )
        raw = bench.strict_json_loads(launch_path.read_bytes())

        def parsed(value: dict) -> bench.ServerLaunchConfig:
            return bench.validate_server_launch_config_value(
                value,
                config_directory=launch_path.parent,
                label="fixture",
                require_local_paths=False,
            )

        wrong_script = json.loads(json.dumps(raw))
        wrong_script["command"][1] = "scripts/run-serving-benchmark-campaign.py"
        with self.assertRaisesRegex(bench.BenchmarkError, "tracked scripts/vllm_teacher"):
            bench.validate_vllm_owned_launch(parsed(wrong_script))

        detached = json.loads(json.dumps(raw))
        index = detached["command"].index("--process-group-mode=inherited")
        detached["command"][index] = "--process-group-mode=detached"
        with self.assertRaisesRegex(bench.BenchmarkError, "process-group-mode=inherited"):
            bench.validate_vllm_owned_launch(parsed(detached))

        duplicate = json.loads(json.dumps(raw))
        boundary = duplicate["command"].index("--")
        duplicate["command"].insert(boundary, "--served-model-id=other")
        with self.assertRaisesRegex(bench.BenchmarkError, "exactly one --served-model-id"):
            bench.validate_vllm_owned_launch(parsed(duplicate))

        with self.assertRaisesRegex(bench.BenchmarkError, "max_model_len disagrees"):
            manifest = valid_vllm_manifest("Qwen3.5-4B")
            manifest["identity"]["max_model_len"] = 16_384
            bench.validate_vllm_owned_launch(parsed(raw), manifest)

    def test_model_fingerprint_read_rate_is_bounded(self) -> None:
        self.assertEqual(
            bench.parse_args([]).model_fingerprint_read_mib_per_second,
            0,
        )
        for value in (0, 64, 256, 16_384):
            with self.subTest(value=value):
                args = bench.parse_args(
                    [f"--model-fingerprint-read-mib-per-second={value}"]
                )
                self.assertEqual(
                    args.model_fingerprint_read_mib_per_second,
                    value,
                )
        for value in (-1, 63, 16_385):
            with self.subTest(value=value), redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                bench.parse_args(
                    [f"--model-fingerprint-read-mib-per-second={value}"]
                )

    def test_memory_source_selectors_are_typed_and_exclusive(self) -> None:
        drm = bench.parse_args(["--memory-path=/tmp/fixture-drm-counter"])
        self.assertEqual(drm.memory_source, "drm")
        nvml = bench.parse_args(["--memory-device-index=1"])
        self.assertEqual(nvml.memory_source, "nvml")
        nvml_uuid = bench.parse_args(
            ["--memory-device-uuid=GPU-01234567-89ab-cdef-0123-456789abcdef"]
        )
        self.assertEqual(nvml_uuid.memory_source, "nvml")
        macos = bench.parse_args(["--memory-source=macos"])
        self.assertEqual(macos.memory_source, "macos")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(
                [
                    "--memory-source=drm",
                    "--memory-device-index=0",
                    "--memory-device-uuid=GPU-01234567-89ab-cdef-0123-456789abcdef",
                ]
            )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(
                [
                    "--memory-source=nvml",
                    "--memory-path=/tmp/fixture-drm-counter",
                ]
            )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(
                [
                    "--memory-source=macos",
                    "--memory-device-index=0",
                ]
            )

    def _run_cli_fixture(
        self,
        fake: FakeServer,
        directory: str,
        *,
        fetch_json: object | None = None,
        engine: str = "kiln",
        extra_args: list[str] | None = None,
        memory_counter_path: Path | None = None,
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
        memory_counter = memory_counter_path or Path(directory) / "vram-used"
        memory_counter.write_text("1024")
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

        def fixture_fingerprint(
            _model_path: Path,
            _model_id: str | None,
            *,
            max_read_mib_per_second: int | None = None,
            **_kwargs: object,
        ) -> dict:
            self.assertIsNone(max_read_mib_per_second)
            return model_fingerprint

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    bench, "repository_identity", return_value=clean_repository
                )
            )
            stack.enter_context(
                mock.patch.object(
                    bench,
                    "fingerprint_model",
                    side_effect=fixture_fingerprint,
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
            provided_extra_args = extra_args or []
            owns_server = any(
                argument == "--server-launch-config"
                or argument.startswith("--server-launch-config=")
                for argument in provided_extra_args
            )
            ownership_args = [] if owns_server else ["--server-pid", "4321"]
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
                    "--prompt-set-id",
                    "cli-prompt-set-v1",
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
                    *ownership_args,
                ]
            arguments.extend(provided_extra_args)
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

    def test_prompt_set_identity_is_stable_across_unique_run_ids(self) -> None:
        left = bench.parse_args(
            [
                "--run-id",
                "candidate-a-v1",
                "--prompt-set-id",
                "shared-prompts-v1",
                "--sizes",
                "8",
            ]
        )
        right = bench.parse_args(
            [
                "--run-id",
                "candidate-b-v1",
                "--prompt-set-id",
                "shared-prompts-v1",
                "--sizes",
                "8",
            ]
        )
        phase = "measure-c008-r000"
        left_prompts = [
            bench.deterministic_prompt(left.prompt_set_id, phase, index)
            for index in range(8)
        ]
        right_prompts = [
            bench.deterministic_prompt(right.prompt_set_id, phase, index)
            for index in range(8)
        ]
        self.assertNotEqual(left.run_id, right.run_id)
        self.assertEqual(left_prompts, right_prompts)
        self.assertEqual(
            bench.deterministic_prompt_set_sha256(
                left.prompt_set_id, phase, 8, "greedy-short"
            ),
            bench.deterministic_prompt_set_sha256(
                right.prompt_set_id, phase, 8, "greedy-short"
            ),
        )
        left_workload = bench.workload_contract(left, [8])
        right_workload = bench.workload_contract(right, [8])
        self.assertNotEqual(
            bench.canonical_sha256(left_workload),
            bench.canonical_sha256(right_workload),
        )
        self.assertEqual(
            bench.workload_fingerprint(left_workload),
            bench.workload_fingerprint(right_workload),
        )
        self.assertNotEqual(
            bench.owned_server_log_path(Path("logs"), left.run_id),
            bench.owned_server_log_path(Path("logs"), right.run_id),
        )

    def test_unique_run_ids_emit_identical_prompt_and_token_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_root = root / "left"
            right_root = root / "right"
            left_root.mkdir()
            right_root.mkdir()
            with FakeServer() as left_server:
                left_code, left_path = self._run_cli_fixture(
                    left_server,
                    left_root,
                    extra_args=[
                        "--run-id",
                        "candidate-left-v1",
                        "--prompt-set-id",
                        "shared-evidence-v1",
                    ],
                )
                left_prompts = [
                    body["messages"][0]["content"] for body in left_server.state.bodies
                ]
            with FakeServer() as right_server:
                right_code, right_path = self._run_cli_fixture(
                    right_server,
                    right_root,
                    extra_args=[
                        "--run-id",
                        "candidate-right-v1",
                        "--prompt-set-id",
                        "shared-evidence-v1",
                    ],
                )
                right_prompts = [
                    body["messages"][0]["content"] for body in right_server.state.bodies
                ]
            left = bench.strict_json_loads(left_path.read_bytes())
            right = bench.strict_json_loads(right_path.read_bytes())

        self.assertEqual((left_code, right_code), (0, 0))
        self.assertNotEqual(
            left["workload"]["run_id"], right["workload"]["run_id"]
        )
        self.assertEqual(left_prompts, right_prompts)
        self.assertEqual(
            left["runs"][0]["prompt_set_sha256"],
            right["runs"][0]["prompt_set_sha256"],
        )
        self.assertEqual(
            left["runs"][0]["prompt_token_counts"],
            right["runs"][0]["prompt_token_counts"],
        )
        self.assertEqual(left["workload_fingerprint"], right["workload_fingerprint"])

    def test_prompt_set_identity_alone_changes_prompts(self) -> None:
        left = bench.parse_args(
            ["--run-id", "same-run-v1", "--prompt-set-id", "prompts-a-v1"]
        )
        right = bench.parse_args(
            ["--run-id", "same-run-v1", "--prompt-set-id", "prompts-b-v1"]
        )
        self.assertEqual(left.run_id, right.run_id)
        self.assertNotEqual(
            bench.deterministic_prompt(left.prompt_set_id, "phase", 0),
            bench.deterministic_prompt(right.prompt_set_id, "phase", 0),
        )

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(["--prompt-set-id", "not portable"])

    def test_measured_cli_requires_explicit_prompt_set_identity(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            return_code = bench.main([])
        self.assertEqual(return_code, 2)
        self.assertIn("explicit --run-id and --prompt-set-id", stderr.getvalue())

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            return_code = bench.main(
                [
                    "--run-id",
                    "coupled-identity-v1",
                    "--prompt-set-id",
                    "coupled-identity-v1",
                ]
            )
        self.assertEqual(return_code, 2)
        self.assertIn("must be distinct", stderr.getvalue())

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
        self.assertEqual(bench.DRIVER_VERSION, "27")
        self.assertEqual(
            bench.PROMPT_TEMPLATE_VERSION, "fixed-serving-profiles-v2"
        )
        self.assertEqual(
            bench.FIXED_PROMPT_TEMPLATE_V2_DRIVER_VERSIONS,
            {"26", "27"},
        )
        self.assertEqual(bench.LONG_PROMPT_REPETITIONS, 61)
        self.assertEqual(bench.LONG_PROMPT_REPETITIONS_V1, 64)
        qualified_prompt_hashes = {
            profile: bench.text_sha256(
                bench.deterministic_prompt(
                    f"cuda-rtx4090-laptop-performance-v1-{profile}",
                    "measure-c064-r000",
                    63,
                    profile,
                )
            )
            for profile in ("long-prefill", "prefix-hit", "mixed")
        }
        self.assertEqual(
            qualified_prompt_hashes,
            {
                "long-prefill": (
                    "sha256:107ee7303c2e3bfc91b7c111bbceb3c9a56db5c4d30682ec94e4e6574d675561"
                ),
                "prefix-hit": (
                    "sha256:b952c83c6c3c66da690781f5644b8d4077c1380742aa66a210449cb46eb3d668"
                ),
                "mixed": (
                    "sha256:982ad94da2170bc57d6f607018b1690b16618e86db2c956b1a278a69722ff576"
                ),
            },
        )
        historical_prompt_set_hashes = {
            profile: bench.deterministic_prompt_set_sha256(
                f"cuda-rtx4090-laptop-performance-v1-{profile}",
                "measure-c001-r000",
                1,
                profile,
                long_prompt_repetitions=bench.LONG_PROMPT_REPETITIONS_V1,
            )
            for profile in ("long-prefill", "prefix-hit")
        }
        self.assertEqual(
            historical_prompt_set_hashes,
            {
                "long-prefill": (
                    "sha256:e8490e64330d2efa4288e173b238ac4df81afa9faebf9f315270c6f98a739fd9"
                ),
                "prefix-hit": (
                    "sha256:5bfd3abdce210aba417f0d7787fc777a5f89765bf4220a7e7bb0f1be2d39a795"
                ),
            },
        )

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
                    "--prompt-set-id",
                    "fixture-prompts-v1",
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
        self.assertTrue(checks["rocm_graph_capture_parity_accounted"])
        parity = result["server"]["rocm_graphs"]["capture_parity"]
        self.assertEqual(parity["batched_capture_successes"], 1)
        self.assertEqual(parity["capture_parity_passes"], 1)
        self.assertEqual(parity["capture_parity_compared_bytes"], 4096)
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

    def test_vllm_stream_requires_matching_runtime_manifest_fingerprint(self) -> None:
        with FakeServer() as fake:
            args = bench.parse_args(
                [
                    "--base-url",
                    fake.base_url,
                    "--engine",
                    "vllm",
                    "--model",
                    "test-model",
                    "--runtime-identity",
                    "test-runtime",
                    "--run-id",
                    "fixture-vllm-identity-v1",
                    "--prompt-set-id",
                    "fixture-vllm-identity-prompts-v1",
                    "--sizes",
                    "1",
                    "--max-tokens",
                    "1",
                    "--warmup-requests",
                    "0",
                ]
            )
            sampler = bench.MemorySampler(None, 50)
            matching = bench.run_once(
                args=args,
                concurrency=1,
                repeat=0,
                max_tokens=1,
                phase="warmup-c001",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                sampler=sampler,
                diagnostics_url=None,
                expected_system_fingerprint="fixture-system-fingerprint",
            )
            mismatching = bench.run_once(
                args=args,
                concurrency=1,
                repeat=0,
                max_tokens=1,
                phase="warmup-c001",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                sampler=sampler,
                diagnostics_url=None,
                expected_system_fingerprint="different-system-fingerprint",
            )
            fake.state.system_fingerprint = None
            missing = bench.run_once(
                args=args,
                concurrency=1,
                repeat=0,
                max_tokens=1,
                phase="warmup-c001",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                sampler=sampler,
                diagnostics_url=None,
                expected_system_fingerprint="fixture-system-fingerprint",
            )

        self.assertEqual(matching["verdict"], "passed")
        self.assertEqual(mismatching["verdict"], "failed")
        self.assertIn(
            "system_fingerprint disagrees",
            mismatching["errors"][0]["error"],
        )
        self.assertEqual(missing["verdict"], "failed")
        self.assertIn(
            "did not report the vLLM runtime manifest system_fingerprint",
            missing["errors"][0]["error"],
        )

    def test_actor_only_diagnostics_require_the_engine_snapshot(self) -> None:
        health = FakeState().health()
        health["decode_runtime"]["batching_engine"] = None
        with self.assertRaisesRegex(
            bench.BenchmarkError, "batching_engine is unavailable"
        ):
            bench.server_diagnostics_snapshot(health)

    def test_driver_v9_server_diagnostics_remain_strict_valid(self) -> None:
        state = FakeState()
        before = bench.server_diagnostics_snapshot(state.health())
        state.request_counters["ok"] = 1
        after = bench.server_diagnostics_snapshot(state.health())
        server = bench.server_diagnostics_delta(before, after)
        legacy = {key: value for key, value in server.items() if key != "rocm_graphs"}
        legacy["schema"] = bench.SERVER_DIAGNOSTICS_SCHEMA_V2
        legacy["routing"] = {
            "batching_actor_effective": True,
            "direct_decode_rendezvous": {
                "scope": "retired",
                "backend_available": True,
                "backend_unavailable_reason": None,
                "actor_active": True,
                "worker_active": False,
                "route_available": False,
            },
        }
        legacy["decode_batcher"] = None
        legacy["batching_engine"] = {
            key: value
            for key, value in legacy["batching_engine"].items()
            if key
            not in {
                "actor_cycle_idle_ms",
                "actor_cycle_idle_source",
                "actor_cycle_idle_active_end",
                "actor_cycle_idle_count",
                "actor_cycle_idle_seconds",
                "process_max_actor_cycle_idle_ms",
            }
        }

        bench.validate_server_diagnostics_v2(legacy, "driver-v9 fixture")

    def test_driver_v13_actor_cycle_idle_is_strictly_accounted(self) -> None:
        state = FakeState()
        state.actor_cycle_idle_ms = 100
        state.actor_cycle_idle_source = "config_file"
        before = bench.server_diagnostics_snapshot(state.health())
        state.request_counters["ok"] = 1
        state.actor_cycle_idle_count = 4
        state.total_actor_cycle_idle_ms = 401.5
        state.max_actor_cycle_idle_ms = 101.25
        after = bench.server_diagnostics_snapshot(state.health())
        server = bench.server_diagnostics_delta(before, after)

        batching = server["batching_engine"]
        self.assertEqual(batching["actor_cycle_idle_ms"], 100)
        self.assertEqual(batching["actor_cycle_idle_source"], "config_file")
        self.assertFalse(batching["actor_cycle_idle_active_end"])
        self.assertEqual(batching["actor_cycle_idle_count"], 4)
        self.assertAlmostEqual(batching["actor_cycle_idle_seconds"], 0.4015)
        self.assertEqual(batching["process_max_actor_cycle_idle_ms"], 101.25)
        self.assertTrue(bench.server_actor_cycle_idle_accounted(server))
        legacy = as_v5_server_diagnostics(server)
        legacy["schema"] = bench.SERVER_DIAGNOSTICS_SCHEMA_V4
        legacy_fallbacks = legacy["rocm_graphs"]["fallbacks"]
        multi_row_count = legacy_fallbacks.pop("multi_row_batch_unsupported")
        legacy_fallbacks["total"] -= multi_row_count
        bench.validate_server_diagnostics_v4(legacy, "driver-v13 fixture")

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

    def test_rocm_graph_parity_allows_pass_before_later_admission_failure(
        self,
    ) -> None:
        state = FakeState()
        state.rocm_graph_parity_counters.update(
            batched_capture_attempts=1,
            batched_capture_failures=1,
            capture_parity_checks=1,
            capture_parity_passes=1,
            capture_parity_compared_bytes=4096,
            capture_parity_duration_micros=10,
        )
        bench.server_diagnostics_snapshot(state.health())

        state.rocm_graph_parity_counters.update(
            batched_capture_successes=1,
            batched_capture_failures=0,
            capture_parity_checks=0,
            capture_parity_passes=0,
            capture_parity_compared_bytes=0,
            capture_parity_duration_micros=0,
        )
        with self.assertRaisesRegex(
            bench.BenchmarkError, "successful batched captures lack parity admission"
        ):
            bench.server_diagnostics_snapshot(state.health())

    def test_rocm_graph_fallback_fails_execution_accounting_gate(self) -> None:
        state = FakeState()
        before = bench.server_diagnostics_snapshot(state.health())
        state.rocm_graph_counters["capture_attempts"] = 1
        state.rocm_graph_counters["capture_deferrals"] = 1
        state.rocm_graph_fallbacks["total"] = 1
        state.rocm_graph_fallbacks["cold_cache_host_round_trip"] = 1
        after = bench.server_diagnostics_snapshot(state.health())
        server = bench.server_diagnostics_delta(before, after)

        bench.validate_server_diagnostics_v5(
            as_v5_server_diagnostics(server), "fallback fixture"
        )
        self.assertFalse(bench.server_rocm_graph_execution_accounted(server))

    def test_multi_row_graph_bypass_is_explicit_and_fails_execution_gate(self) -> None:
        state = FakeState()
        before = bench.server_diagnostics_snapshot(state.health())
        state.max_active = 4
        state.counters["total_decode_forwards"] = 1
        state.counters["total_batched_decode_forwards"] = 1
        state.counters["total_decode_rows"] = 4
        state.counters["total_decode_tokens"] = 4
        state.rocm_graph_fallbacks["total"] = 1
        state.rocm_graph_fallbacks["multi_row_batch_unsupported"] = 1
        state.rocm_graph_fallbacks["slow"] = 1
        state.rocm_graph_fallbacks["total_duration_micros"] = 130_000
        state.rocm_graph_fallbacks["max_duration_micros"] = 130_000
        after = bench.server_diagnostics_snapshot(state.health())
        server = bench.server_diagnostics_delta(before, after)

        bench.validate_server_diagnostics_v5(
            as_v5_server_diagnostics(server), "multi-row fixture"
        )
        self.assertEqual(
            server["rocm_graphs"]["fallbacks"][
                "multi_row_batch_unsupported"
            ],
            1,
        )
        self.assertFalse(bench.server_rocm_graph_execution_accounted(server))

    def test_multi_row_graph_fallback_requires_measured_batched_forward(self) -> None:
        state = FakeState()
        before = bench.server_diagnostics_snapshot(state.health())
        state.max_active = 4
        state.rocm_graph_fallbacks["total"] = 1
        state.rocm_graph_fallbacks["multi_row_batch_unsupported"] = 1
        after = bench.server_diagnostics_snapshot(state.health())
        server = bench.server_diagnostics_delta(before, after)

        with self.assertRaisesRegex(
            bench.BenchmarkError, "without a measured multi-row batching route"
        ):
            bench.validate_server_diagnostics_v5(
                as_v5_server_diagnostics(server), "contradictory fixture"
            )

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
            *bench.ROCM_GRAPH_BATCHED_CAPTURE_COUNTER_FIELDS,
            *bench.ROCM_GRAPH_CAPTURE_PARITY_COUNTER_FIELDS,
            *bench.ROCM_GRAPH_GAUGE_FIELDS,
            "fallbacks",
        ):
            graph[field] = None

        before = bench.server_diagnostics_snapshot(health)
        server = bench.server_diagnostics_delta(before, before)

        bench.validate_server_diagnostics_v5(
            as_v5_server_diagnostics(server), "unavailable fixture"
        )
        self.assertTrue(bench.server_rocm_graph_execution_accounted(server))
        self.assertIsNone(server["rocm_graphs"]["replay_successes"])

        server["rocm_graphs"]["state"] = "busy"
        server["rocm_graphs"]["unavailable_reason"] = "graph_runner_busy"
        bench.validate_server_diagnostics_v5(
            as_v5_server_diagnostics(server), "busy fixture"
        )
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

    def test_request_performance_evidence_must_be_index_ordered(self) -> None:
        performance = valid_request_performance(3)
        evidence = [
            {"index": index, "performance": json.loads(json.dumps(performance))}
            for index in (1, 0)
        ]
        summary = bench.build_request_phase_summary(
            row["performance"] for row in evidence
        )
        output_evidence = [
            {"index": index, "completion_tokens": 3, "finish_reason": "length"}
            for index in (0, 1)
        ]
        with self.assertRaisesRegex(bench.BenchmarkError, "ordered"):
            bench.validate_request_performance_evidence(
                evidence,
                summary,
                label="fixture.request_performance",
                engine_name="kiln",
                concurrency=2,
                success_count=2,
                error_indices=set(),
                prompt_token_counts=[42, 42],
                output_evidence_rows=output_evidence,
                completion_tokens=6,
            )

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
            "reference_role": "qualification_gate",
            "workload_fingerprint": "sha256:workload",
            "workload": {"comparison_mode": "exact_output"},
            "engine": {"model_identity": {"content_sha256": "sha256:model"}},
            "memory_sampler": {
                "source": "drm_vram_used",
                "path": "/sys/class/drm/card0/device/mem_info_vram_used",
                "device": None,
                "interval_ms": 50,
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
        reference["driver_version"] = bench.DRIVER_VERSION
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

                path.write_text('{"schema":"first","schema":"second"}')
                with self.assertRaisesRegex(
                    bench.BenchmarkError, "duplicate JSON object key"
                ):
                    bench.compare_reference(current, path)

    def test_same_artifact_graph_discriminator_separates_reproducibility(self) -> None:
        state = FakeState()
        before = bench.server_diagnostics_snapshot(state.health())
        state.max_active = 4
        state.counters["total_decode_forwards"] = 1
        state.counters["total_batched_decode_forwards"] = 1
        state.counters["total_decode_rows"] = 4
        state.counters["total_decode_tokens"] = 4
        state.rocm_graph_counters.update(
            capture_attempts=1,
            capture_successes=1,
            cache_admission_successes=1,
            replay_attempts=3,
            replay_successes=3,
        )
        state.rocm_graph_parity_counters.update(
            batched_capture_attempts=1,
            batched_capture_successes=1,
            capture_parity_checks=1,
            capture_parity_passes=1,
            capture_parity_compared_bytes=4096,
            capture_parity_duration_micros=10,
        )
        state.rocm_graph_gauges.update(
            captured_graph_count=1,
            graph_slot_count=1,
            idle_graph_slot_count=1,
        )
        candidate_server = bench.server_diagnostics_delta(
            before, bench.server_diagnostics_snapshot(state.health())
        )
        eager_server = json.loads(json.dumps(candidate_server))
        eager_graph = eager_server["rocm_graphs"]
        eager_graph["capture_requested"] = False
        eager_graph["capture_enabled"] = False
        for field in bench.ROCM_GRAPH_COUNTER_FIELDS:
            eager_graph[field] = 0
        eager_graph["fallbacks"] = {
            field: 0 for field in eager_graph["fallbacks"]
        }
        eager_graph["capture_parity"] = {
            field: 0 for field in eager_graph["capture_parity"]
        }

        output = {
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
        engine = {
            "name": "kiln",
            "runtime_identity": "kiln-git:fixture",
            "runtime_artifact": {"sha256": "sha256:" + "d" * 64},
            "model_identity": {"content_sha256": "sha256:model"},
        }
        current = {
            "schema": bench.SCHEMA,
            "driver_version": bench.DRIVER_VERSION,
            "reference_role": "same_artifact_graph_eager_discriminator",
            "workload_fingerprint": "sha256:workload",
            "workload": {"comparison_mode": "exact_output"},
            "engine": engine,
            "memory_sampler": {
                "source": "drm_vram_used",
                "path": "/sys/class/drm/card0/device/mem_info_vram_used",
                "device": None,
                "interval_ms": 50,
            },
            "runs": [
                {
                    "concurrency": 1,
                    "repeat": 0,
                    "prompt_token_counts": [42],
                    "prompt_set_sha256": "sha256:prompts",
                    "output_set_sha256": "sha256:actual",
                    "output_evidence": [output],
                    "server": candidate_server,
                }
            ],
        }
        reference = json.loads(json.dumps(current))
        reference["reference_role"] = "qualification_gate"
        reference["verdict"] = "passed"
        reference["engine"] = json.loads(json.dumps(engine))
        reference["runs"][0]["server"] = eager_server
        reference["runs"][0]["output_set_sha256"] = "sha256:expected"
        reference["runs"][0]["output_evidence"][0]["output_sha256"] = (
            "sha256:" + "e" * 64
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(reference))
            with mock.patch.object(bench, "validate_benchmark_receipt"):
                comparison = bench.compare_reference(current, path)
                self.assertFalse(comparison["matched"])
                self.assertEqual(comparison["verdict_effect"], "evidence_only")
                self.assertTrue(
                    comparison["reference_execution"][
                        "all_rows_capture_disabled"
                    ]
                )

                wrong_device = json.loads(json.dumps(reference))
                wrong_device["memory_sampler"]["path"] = (
                    "/sys/class/drm/card1/device/mem_info_vram_used"
                )
                path.write_text(json.dumps(wrong_device))
                with self.assertRaisesRegex(
                    bench.BenchmarkError, "different accelerator device"
                ):
                    bench.compare_reference(current, path)

                broken_artifact = json.loads(json.dumps(reference))
                broken_artifact["engine"]["runtime_artifact"]["sha256"] = (
                    "sha256:" + "f" * 64
                )
                path.write_text(json.dumps(broken_artifact))
                with self.assertRaisesRegex(
                    bench.BenchmarkError, "identical runtime artifact"
                ):
                    bench.compare_reference(current, path)

                failed_reference = json.loads(json.dumps(reference))
                failed_reference["verdict"] = "failed"
                path.write_text(json.dumps(failed_reference))
                with self.assertRaisesRegex(
                    bench.BenchmarkError, "passed eager reference"
                ):
                    bench.compare_reference(current, path)

                path.write_text(json.dumps(reference))
                broken_candidate = json.loads(json.dumps(current))
                broken_candidate["runs"][0]["server"]["rocm_graphs"][
                    "capture_parity"
                ]["capture_parity_passes_end"] = 0
                with self.assertRaisesRegex(
                    bench.BenchmarkError, "lacks measured graph parity evidence"
                ):
                    bench.compare_reference(broken_candidate, path)

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

    def test_owned_server_readiness_retries_transient_probe_failures(self) -> None:
        class Process:
            @staticmethod
            def poll() -> None:
                return None

        class Config:
            startup_timeout_seconds = 1.0
            readiness_poll_interval_seconds = 0.001

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text("", encoding="ascii")
            server = bench.OwnedServer(
                process=Process(),
                identity=None,
                config=Config(),
                log_path=log,
                log_handle=None,
            )
            with mock.patch.object(
                bench,
                "probe_models",
                side_effect=[
                    bench.BenchmarkError("connection refused"),
                    ["test-model"],
                ],
            ) as probe:
                models = bench.wait_for_owned_server_models(
                    server,
                    "http://127.0.0.1:8420",
                    {},
                )

        self.assertEqual(models, ["test-model"])
        self.assertEqual(probe.call_count, 2)

    def test_shutdown_accounting_failure_still_drains_owned_group(self) -> None:
        executable = (
            bench._darwin_process_identity(os.getpid())[2]
            if sys.platform == "darwin"
            else sys.executable
        )
        command = (
            executable,
            "-c",
            "import time; time.sleep(60)",
        )
        process = bench.subprocess.Popen(
            command,
            stdin=bench.subprocess.DEVNULL,
            stdout=bench.subprocess.DEVNULL,
            stderr=bench.subprocess.DEVNULL,
            start_new_session=True,
        )
        if sys.platform == "darwin":
            time.sleep(0.05)
        identity = bench.AttachedProcessGroup.attach(
            process.pid,
            expected_command=command if sys.platform == "darwin" else None,
        )

        class Config:
            shutdown_timeout_seconds = 1.0
            acceptable_exit_codes = (0,)

        server = bench.OwnedServer(
            process=process,
            identity=identity,
            config=Config(),
            log_path=Path("/unused"),
            log_handle=None,
        )
        with (
            mock.patch.object(
                bench,
                "_wait_for_owned_process",
                side_effect=bench.BenchmarkError("invalid timeout accounting"),
            ),
            self.assertRaisesRegex(
                bench.OwnedServerShutdownError,
                "invalid timeout accounting",
            ) as raised,
        ):
            bench.shutdown_owned_server(server)

        self.assertIsNotNone(process.poll())
        self.assertFalse(bench.process_group_alive(process.pid))
        self.assertTrue(raised.exception.shutdown["forced"])
        self.assertEqual(
            raised.exception.shutdown["returncode"],
            process.returncode,
        )
        self.assertFalse(
            raised.exception.shutdown["process_group_alive_end"],
        )

    def test_zombie_only_process_group_is_execution_quiescent(self) -> None:
        if sys.platform != "linux" or not Path("/proc/self/stat").is_file():
            self.skipTest("Linux procfs process states are required")
        process = bench.subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdin=bench.subprocess.DEVNULL,
            stdout=bench.subprocess.DEVNULL,
            stderr=bench.subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 5.0
            while True:
                state, process_group_id, _start = bench.AttachedProcessGroup._read_stat(
                    process.pid,
                    Path("/proc"),
                )
                if state == "Z":
                    break
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.005)
            self.assertEqual(process_group_id, process.pid)
            bench.os.killpg(process.pid, 0)
            self.assertFalse(bench.process_group_alive(process.pid))
        finally:
            process.wait()

    def test_process_group_liveness_fails_closed_on_proc_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc_root = Path(directory)
            (proc_root / "101").mkdir()
            stat_path = proc_root / "101" / "stat"
            with (
                mock.patch.object(bench.os, "killpg"),
                mock.patch.object(bench.sys, "platform", "linux"),
            ):
                stat_path.write_text("101 (fixture) Z 1 101 101 0\n", encoding="ascii")
                self.assertFalse(bench.process_group_alive(101, proc_root))
                stat_path.write_text("101 (fixture) S 1 101 101 0\n", encoding="ascii")
                self.assertTrue(bench.process_group_alive(101, proc_root))
                stat_path.write_text("malformed\n", encoding="ascii")
                self.assertTrue(bench.process_group_alive(101, proc_root))
                stat_path.unlink()
                self.assertTrue(bench.process_group_alive(101, proc_root))

    def test_owned_server_log_retries_transient_fsync_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            handle = path.open("wb")
            handle.write(b"complete server log\n")
            server = bench.OwnedServer(
                process=None,
                identity=None,
                config=None,
                log_path=path,
                log_handle=handle,
            )
            real_fsync = bench.os.fsync
            attempts = 0

            def flaky_fsync(fd: int) -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise BlockingIOError(bench.errno.EAGAIN, "injected transient")
                real_fsync(fd)

            with mock.patch.object(bench.os, "fsync", side_effect=flaky_fsync):
                log = bench.close_owned_server_log(server)

        self.assertEqual(attempts, 2)
        self.assertTrue(handle.closed)
        self.assertEqual(log["bytes"], len(b"complete server log\n"))

    def test_owned_server_log_retains_identity_after_durability_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            handle = path.open("wb")
            handle.write(b"readable server log\n")
            server = bench.OwnedServer(
                process=None,
                identity=None,
                config=None,
                log_path=path,
                log_handle=handle,
            )
            with (
                mock.patch.object(
                    bench.os,
                    "fsync",
                    side_effect=OSError(bench.errno.EIO, "injected durability failure"),
                ),
                self.assertRaisesRegex(
                    bench.OwnedServerLogError,
                    "injected durability failure",
                ) as raised,
            ):
                bench.close_owned_server_log(server)

        self.assertTrue(handle.closed)
        self.assertEqual(
            raised.exception.log["bytes"],
            len(b"readable server log\n"),
        )
        self.assertRegex(
            raised.exception.log["sha256"],
            r"\Asha256:[0-9a-f]{64}\Z",
        )

    def test_owned_server_finalizer_retains_independent_failure_evidence(self) -> None:
        shutdown = {
            "signal": "SIGTERM",
            "signal_sent": True,
            "forced": True,
            "returncode": 0,
            "acceptable_exit_codes": [0],
            "elapsed_seconds": 0.5,
            "process_group_alive_end": False,
        }
        log = {
            "path": "/tmp/server.log",
            "bytes": 42,
            "sha256": "sha256:" + "a" * 64,
        }
        with (
            mock.patch.object(
                bench,
                "shutdown_owned_server",
                side_effect=bench.OwnedServerShutdownError(
                    "injected shutdown accounting failure",
                    shutdown,
                ),
            ),
            mock.patch.object(
                bench,
                "close_owned_server_log",
                side_effect=bench.OwnedServerLogError(
                    "injected log durability failure",
                    log,
                ),
            ),
        ):
            observed_shutdown, observed_log, failures = bench.finalize_owned_server(
                mock.Mock()
            )

        self.assertEqual(observed_shutdown, shutdown)
        self.assertEqual(observed_log, log)
        self.assertEqual(
            [type(failure) for failure in failures],
            [bench.OwnedServerShutdownError, bench.OwnedServerLogError],
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
            with mock.patch.object(bench.sys, "platform", "linux"):
                attached = bench.AttachedProcessGroup.attach(4321, proc_root=root)
            self.assertIsNone(attached.poll())
            self.assertEqual(attached.receipt_identity()["process_group_id"], 4321)

            write_stat(4321, 999999)
            self.assertEqual(attached.poll(), 0)
            write_stat(4000, 123456)
            with (
                mock.patch.object(bench.sys, "platform", "linux"),
                self.assertRaisesRegex(
                    bench.BenchmarkError, "lead its process group"
                ),
            ):
                bench.AttachedProcessGroup.attach(4321, proc_root=root)

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
            python_executable = (
                bench._darwin_process_identity(os.getpid())[2]
                if sys.platform == "darwin"
                else sys.executable
            )
            launch_value = {
                "schema": bench.SERVER_LAUNCH_SCHEMA,
                "id": "fixture-owned-server-v1",
                "command": [python_executable, "./fixture-server.py", str(port)],
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

            def flaky_attach(
                pid: int,
                *,
                expected_command: tuple[str, ...] | None = None,
            ) -> object:
                nonlocal attach_attempts
                attach_attempts += 1
                if attach_attempts == 1:
                    raise bench.BenchmarkError("injected pre-exec identity race")
                return original_attach(pid, expected_command=expected_command)

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
                "log": log,
                "shutdown": shutdown,
            }
            mode, passed = bench.validate_server_lifecycle(lifecycle)
            self.assertEqual(mode, "owned_process_group")
            self.assertTrue(passed)

    def test_unpaced_owned_server_exit_retains_complete_failed_receipt(self) -> None:
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
                        "command": [
                            str(executable),
                            "scripts/vllm_teacher.py",
                            f"--model-path={root / 'model'}",
                            "--served-model-id=test-model",
                            "--process-group-mode=inherited",
                            f"--snapshot-root={root / 'snapshots'}",
                            f"--cache-root={root / 'caches'}",
                            "--max-top-k=20",
                            "--max-model-len=32768",
                            "--",
                        ],
                        "working_directory": str(ROOT),
                        "log_directory": str(root / "logs"),
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
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        bench, "repository_identity", return_value=clean_repository
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        bench,
                        "fingerprint_model",
                        return_value=model_fingerprint,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        bench,
                        "_fsync_owned_server_log",
                        side_effect=OSError(
                            bench.errno.EIO,
                            "injected owned log durability failure",
                        ),
                    )
                )
                if sys.platform == "darwin":
                    def attach_fixture(
                        pid: int,
                        **_kwargs,
                    ) -> bench.AttachedProcessGroup:
                        process_group_id, start_time, observed_executable = (
                            bench._darwin_process_identity(pid)
                        )
                        return bench.AttachedProcessGroup(
                            pid=pid,
                            process_group_id=process_group_id,
                            start_time_ticks=start_time,
                            boot_id="fixture-darwin-boot",
                            executable=observed_executable,
                            cmdline_sha256="sha256:" + "a" * 64,
                            platform_name="darwin",
                        )

                    stack.enter_context(
                        mock.patch.object(
                            bench.AttachedProcessGroup,
                            "attach",
                            side_effect=attach_fixture,
                        )
                    )
                return_code = bench.main(
                    [
                        "--engine=vllm",
                        f"--base-url=http://127.0.0.1:{port}",
                        "--model=test-model",
                        f"--model-path={root / 'model'}",
                        "--runtime-identity=test-vllm-runtime",
                        f"--runtime-artifact={runtime_artifact}",
                        "--run-id=owned-startup-failure-v1",
                        "--prompt-set-id=owned-startup-prompts-v1",
                        "--sizes=1",
                        "--max-tokens=1",
                        "--warmup-requests=0",
                        f"--memory-path={memory_counter}",
                        "--memory-limit-bytes=2048",
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
            self.assertIn(
                "injected owned log durability failure",
                receipt["completion"]["failures"][1]["detail"],
            )
            lifecycle = receipt["server_lifecycle"]
            self.assertEqual(lifecycle["shutdown"]["returncode"], 1)
            self.assertFalse(lifecycle["shutdown"]["signal_sent"])
            self.assertFalse(lifecycle["shutdown"]["forced"])
            self.assertFalse(lifecycle["shutdown"]["process_group_alive_end"])
            self.assertGreater(lifecycle["log"]["bytes"], 0)

    def test_cli_writes_a_self_hashing_passed_receipt(self) -> None:
        with FakeServer() as fake, tempfile.TemporaryDirectory() as directory:
            return_code, output = self._run_cli_fixture(fake, directory)
            receipt = bench.strict_json_loads(output.read_bytes())
            self.assertEqual(bench.main(["--validate-receipt", str(output)]), 0)
            self.assertEqual(receipt["driver_version"], bench.DRIVER_VERSION)
            self.assertEqual(
                receipt["memory_sampler"],
                {
                    "source": "drm_vram_used",
                    "path": str((Path(directory) / "vram-used").resolve()),
                    "device": None,
                    "interval_ms": 50,
                },
            )
            self.assertEqual(receipt["reference_role"], "qualification_gate")
            self.assertEqual(
                receipt["workload"]["prompt_set_id"], "cli-prompt-set-v1"
            )
            run = receipt["runs"][0]
            self.assertEqual(len(run["request_performance"]), 1)
            self.assertEqual(
                run["request_performance"][0]["performance"]["completion_tokens"],
                3,
            )
            self.assertEqual(
                run["request_phase_summary"]["phases"]["actor_cycle_idle_ms"][
                    "observed_request_count"
                ],
                1,
            )
            self.assertTrue(
                next(
                    gate
                    for gate in run["gates"]
                    if gate["name"] == "request_performance_accounted"
                )["passed"]
            )
            missing_discriminator = json.loads(json.dumps(receipt))
            missing_discriminator["reference_role"] = (
                "same_artifact_graph_eager_discriminator"
            )
            missing_discriminator.pop("receipt_sha256")
            missing_discriminator["receipt_sha256"] = bench.canonical_sha256(
                missing_discriminator
            )
            with self.assertRaisesRegex(
                bench.BenchmarkError, "requires a reference comparison"
            ):
                bench.validate_benchmark_receipt(missing_discriminator)

            stale_identity = json.loads(json.dumps(receipt))
            stale_identity["workload"]["prompt_set_id"] = "stale-prompts-v1"
            stale_identity["workload_fingerprint"] = bench.workload_fingerprint(
                stale_identity["workload"]
            )
            stale_identity.pop("receipt_sha256")
            stale_identity["receipt_sha256"] = bench.canonical_sha256(stale_identity)
            with self.assertRaisesRegex(bench.BenchmarkError, "stale"):
                bench.validate_benchmark_receipt(stale_identity)

            missing_identity = json.loads(json.dumps(receipt))
            missing_identity["workload"].pop("prompt_set_id")
            missing_identity.pop("receipt_sha256")
            missing_identity["receipt_sha256"] = bench.canonical_sha256(
                missing_identity
            )
            with self.assertRaisesRegex(bench.BenchmarkError, "keys"):
                bench.validate_benchmark_receipt(missing_identity)

            malformed_identity = json.loads(json.dumps(receipt))
            malformed_identity["workload"]["prompt_set_id"] = "not portable"
            malformed_identity["workload_fingerprint"] = bench.workload_fingerprint(
                malformed_identity["workload"]
            )
            malformed_identity.pop("receipt_sha256")
            malformed_identity["receipt_sha256"] = bench.canonical_sha256(
                malformed_identity
            )
            with self.assertRaisesRegex(bench.BenchmarkError, "portable identifier"):
                bench.validate_benchmark_receipt(malformed_identity)

            coupled_identity = json.loads(json.dumps(receipt))
            coupled_identity["workload"]["prompt_set_id"] = coupled_identity[
                "workload"
            ]["run_id"]
            coupled_identity["workload_fingerprint"] = bench.workload_fingerprint(
                coupled_identity["workload"]
            )
            coupled_identity.pop("receipt_sha256")
            coupled_identity["receipt_sha256"] = bench.canonical_sha256(
                coupled_identity
            )
            with self.assertRaisesRegex(bench.BenchmarkError, "must be distinct"):
                bench.validate_benchmark_receipt(coupled_identity)

            nvml_receipt = json.loads(json.dumps(receipt))
            nvml_receipt["memory_sampler"] = {
                "source": "nvml_used",
                "path": None,
                "device": {
                    "selector": "explicit_uuid",
                    "index": 0,
                    "enumerated_device_count": 2,
                    "uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
                    "name": "NVIDIA GeForce RTX 4090",
                    "total_bytes": 24 * 1024**3,
                    "library": "libnvidia-ml.so.1",
                    "nvml_version": "13.580.65",
                },
                "interval_ms": 50,
            }
            nvml_receipt.pop("receipt_sha256")
            nvml_receipt["receipt_sha256"] = bench.canonical_sha256(nvml_receipt)
            bench.validate_benchmark_receipt(nvml_receipt)

            invalid_nvml = json.loads(json.dumps(nvml_receipt))
            invalid_nvml["memory_sampler"]["device"]["uuid"] = ""
            invalid_nvml.pop("receipt_sha256")
            invalid_nvml["receipt_sha256"] = bench.canonical_sha256(invalid_nvml)
            with self.assertRaisesRegex(bench.BenchmarkError, "uuid is invalid"):
                bench.validate_benchmark_receipt(invalid_nvml)

            macos_receipt = json.loads(json.dumps(receipt))
            macos_receipt["memory_sampler"] = {
                "source": "macos_unified_used",
                "path": None,
                "device": {
                    "selector": "system_default",
                    "index": 0,
                    "enumerated_device_count": 1,
                    "name": "Apple M1",
                    "total_bytes": 16 * 1024**3,
                    "unified_memory": True,
                    "counter": "memory_pressure_free_percentage",
                    "available_definition": (
                        "physical_total_times_reported_free_percentage"
                    ),
                },
                "interval_ms": 250,
            }
            macos_receipt.pop("receipt_sha256")
            macos_receipt["receipt_sha256"] = bench.canonical_sha256(
                macos_receipt
            )
            bench.validate_benchmark_receipt(macos_receipt)

            historical_macos = json.loads(json.dumps(macos_receipt))
            historical_macos["driver_version"] = bench.PREVIOUS_DRIVER_VERSION
            historical_macos.pop("receipt_sha256")
            historical_macos["receipt_sha256"] = bench.canonical_sha256(
                historical_macos
            )
            with self.assertRaisesRegex(
                bench.BenchmarkError, "device-memory source is unsupported"
            ):
                bench.validate_benchmark_receipt(historical_macos)

            tampered_summary = json.loads(json.dumps(receipt))
            tampered_distribution = tampered_summary["runs"][0][
                "request_phase_summary"
            ]["phases"]["actor_cycle_idle_ms"]
            for statistic in ("p50", "p99", "max"):
                tampered_distribution[statistic] += 1.0
            tampered_summary.pop("receipt_sha256")
            tampered_summary["receipt_sha256"] = bench.canonical_sha256(
                tampered_summary
            )
            with self.assertRaisesRegex(bench.BenchmarkError, "not derived"):
                bench.validate_benchmark_receipt(tampered_summary)

            tampered_request = json.loads(json.dumps(receipt))
            tampered_request["runs"][0]["request_performance"][0][
                "performance"
            ]["completion_tokens"] += 1
            tampered_request.pop("receipt_sha256")
            tampered_request["receipt_sha256"] = bench.canonical_sha256(
                tampered_request
            )
            with self.assertRaisesRegex(
                bench.BenchmarkError, "emitted_tokens disagrees"
            ):
                bench.validate_benchmark_receipt(tampered_request)

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
            for row in [vllm_receipt["warmup"], *vllm_receipt["runs"]]:
                if row is None:
                    continue
                row["request_performance"] = None
                row["request_phase_summary"] = None
                row["gates"] = [
                    gate
                    for gate in row["gates"]
                    if gate["name"] != "request_performance_accounted"
                ]
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
                    memory_counter_path=Path(reference_directory) / "vram-used",
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
