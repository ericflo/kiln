#!/usr/bin/env python3
"""Run a source-bound, qualification-grade mixed serving load on ROCm."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import http.client
import json
import math
import os
import re
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "mixed-load"
RESULT_ENV = "KILN_QUALIFICATION_CASE_RESULT"
VARIANT_ENV = "KILN_QUALIFICATION_VARIANT_ID"
MODEL_ID = "Qwen3.5-4B"
BUILD_PACKAGE = "kiln-server"
BUILD_BINARY = "kiln"
BUILD_PROFILE = "release"
BUILD_FEATURES = "rocm"
BUILD_ROCM_PATH = "/opt/rocm"
BUILD_ROCM_ARCHS = "gfx1151"
STARTUP_TIMEOUT_SECONDS = 240.0
REQUEST_TIMEOUT_SECONDS = 120.0
OVERALL_TIMEOUT_SECONDS = 420.0
NORMAL_REQUESTS = 8
NORMAL_MAX_TOKENS = 128
LONG_PREFILL_WORDS = 1536
LONG_PREFILL_MAX_TOKENS = 32
PRESSURE_PEER_PROMPT_WORDS = 64
PRESSURE_PEER_MAX_TOKENS = 512
PRESSURE_PEER_SEED_OFFSET = 103
WARMUP_MAX_TOKENS = 32
MAX_WARMUP_REQUESTS = 4
SLOW_MAX_TOKENS = 4096
CANCELLATION_AFTER_DELTAS = 4
MEMORY_POLL_INTERVAL_SECONDS = 0.5
OUTLIER_ABSOLUTE_MS = 250.0
OUTLIER_MULTIPLIER = 5.0
OUTLIER_HISTORY_SIZE = 64
SLOW_SOCKET_BUFFER_BYTES = 4096
HTTP_SEND_BUFFER_BYTES = 4096
STREAM_STALL_GRACE_MS = 2000
SLO_TTFT_MS = 30_000.0
SLO_E2E_MS = 120_000.0
STREAM_READ_POLL_SECONDS = 0.25


def _variant_config(*, kv_autoscale: bool, rocm_graphs: bool) -> dict[str, Any]:
    return {
        "build": {
            "binary": BUILD_BINARY,
            "features": BUILD_FEATURES,
            "locked": True,
            "no_default_features": True,
            "offline": True,
            "package": BUILD_PACKAGE,
            "profile": BUILD_PROFILE,
            "rocm_archs": BUILD_ROCM_ARCHS,
            "rocm_path": BUILD_ROCM_PATH,
        },
        "runtime": {
            "kv_autoscale_enabled": kv_autoscale,
            "memory_reclaim_mode": "off",
            "rocm_graphs_enabled": rocm_graphs,
        },
        "server": {
            "chat_performance_metadata_enabled": True,
            "debug_endpoints_enabled": True,
            "default_thinking_enabled": False,
            "http_send_buffer_bytes": HTTP_SEND_BUFFER_BYTES,
            "log_format": "json",
            "request_timeout_seconds": 180,
            "stream_stall_grace_ms": STREAM_STALL_GRACE_MS,
        },
        "workload": {
            "cancellation_after_semantic_deltas": CANCELLATION_AFTER_DELTAS,
            "long_prefill_max_tokens": LONG_PREFILL_MAX_TOKENS,
            "long_prefill_words": LONG_PREFILL_WORDS,
            "max_warmup_requests": MAX_WARMUP_REQUESTS,
            "memory_poll_interval_ms": int(MEMORY_POLL_INTERVAL_SECONDS * 1000),
            "normal_max_tokens": NORMAL_MAX_TOKENS,
            "normal_requests": NORMAL_REQUESTS,
            "outlier_absolute_ms": int(OUTLIER_ABSOLUTE_MS),
            "outlier_history_size": OUTLIER_HISTORY_SIZE,
            "outlier_multiplier": int(OUTLIER_MULTIPLIER),
            "overall_timeout_seconds": int(OVERALL_TIMEOUT_SECONDS),
            "pressure_peer_dispatch": "after_slow_headers_before_pressure_wait",
            "pressure_peer_max_tokens": PRESSURE_PEER_MAX_TOKENS,
            "pressure_peer_prompt_words": PRESSURE_PEER_PROMPT_WORDS,
            "pressure_peer_seed_offset": PRESSURE_PEER_SEED_OFFSET,
            "request_timeout_seconds": int(REQUEST_TIMEOUT_SECONDS),
            "slow_socket_buffer_bytes": SLOW_SOCKET_BUFFER_BYTES,
            "slow_max_tokens": SLOW_MAX_TOKENS,
            "startup_timeout_seconds": int(STARTUP_TIMEOUT_SECONDS),
            "warmup_max_tokens": WARMUP_MAX_TOKENS,
        },
    }


VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "default": _variant_config(kv_autoscale=True, rocm_graphs=True),
    "autoscale-off": _variant_config(kv_autoscale=False, rocm_graphs=True),
    "graphs-off": _variant_config(kv_autoscale=True, rocm_graphs=False),
    "both-off": _variant_config(kv_autoscale=False, rocm_graphs=False),
}


METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    "attributed_itl_outlier_count": ("count", "sum", True),
    "batching_batched_decode_forward_count": ("count", "sum", False),
    "batching_decode_forward_count": ("count", "sum", False),
    "batching_decode_row_count": ("rows", "sum", False),
    "batching_max_observed_batch_size": ("rows", "max", False),
    "batching_mean_rows_per_forward": ("rows", "mean", False),
    "batching_total_errors": ("count", "sum", True),
    "cancellation_confirmed_count": ("count", "sum", False),
    "client_backpressure_event_count": ("count", "sum", True),
    "client_backpressure_wait_ms": ("ms", "sum", True),
    "client_stall_eviction_count": ("count", "sum", True),
    "completion_token_count": ("tokens", "sum", False),
    "e2e_latency_ms_p50": ("ms", "p50", True),
    "e2e_latency_ms_p99": ("ms", "p99", True),
    "e2e_latency_ms_p999": ("ms", "p99.9", True),
    "graph_measured_capture_attempt_count": ("count", "sum", False),
    "graph_measured_capture_deferral_count": ("count", "sum", True),
    "graph_measured_capture_failure_count": ("count", "sum", True),
    "graph_measured_capture_success_count": ("count", "sum", False),
    "graph_measured_live_count_end": ("graphs", "exact", False),
    "graph_measured_replay_attempt_count": ("count", "sum", False),
    "graph_measured_replay_failure_count": ("count", "sum", True),
    "graph_measured_replay_success_count": ("count", "sum", False),
    "graph_pre_measurement_capture_success_count": ("count", "exact", False),
    "graph_pre_measurement_failure_count": ("count", "exact", True),
    "graph_pre_measurement_replay_success_count": ("count", "exact", False),
    "itl_ms_p50": ("ms", "p50", True),
    "itl_ms_p99": ("ms", "p99", True),
    "itl_ms_p999": ("ms", "p99.9", True),
    "kv_blocks_end": ("blocks", "exact", False),
    "kv_blocks_start": ("blocks", "exact", False),
    "kv_resize_event_count": ("count", "sum", True),
    "long_prefill_prompt_tokens": ("tokens", "exact", False),
    "memory_reclaim_event_count": ("count", "sum", True),
    "output_token_throughput_per_second": ("tokens/s", "rate", False),
    "peak_gpu_memory_used_bytes": ("bytes", "max", True),
    "prompt_token_count": ("tokens", "sum", False),
    "request_count": ("count", "sum", False),
    "request_failure_count": ("count", "sum", True),
    "request_throughput_per_second": ("requests/s", "rate", False),
    "response_queue_delay_ms_p50": ("ms", "p50", True),
    "response_queue_delay_ms_p99": ("ms", "p99", True),
    "response_queue_delay_ms_p999": ("ms", "p99.9", True),
    "slo_goodput_requests_per_second": ("requests/s", "rate", False),
    "slow_consumer_peer_success_count": ("count", "sum", False),
    "ttft_ms_p50": ("ms", "p50", True),
    "ttft_ms_p99": ("ms", "p99", True),
    "ttft_ms_p999": ("ms", "p99.9", True),
    "unexplained_itl_outlier_count": ("count", "sum", True),
    "zero_token_response_count": ("count", "sum", True),
}

GRAPH_MONOTONIC_FIELDS = (
    "capture_attempts",
    "capture_successes",
    "capture_deferrals",
    "capture_failures",
    "replay_attempts",
    "replay_successes",
    "replay_failures",
    "failures",
)
GRAPH_GAUGE_FIELDS = ("captured_graph_count",)


class QualificationError(RuntimeError):
    pass


def remaining_until(deadline: float, label: str, cap: float | None = None) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{label} exceeded the overall deadline")
    return min(remaining, cap) if cap is not None else remaining


def trace(event: str, **fields: Any) -> None:
    value = {"event": event, **fields}
    print(json.dumps(value, sort_keys=True, separators=(",", ":")), flush=True)


def percentile_r7(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    rank = probability * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class SSEParser:
    """Incremental SSE parser tolerant of arbitrary byte fragmentation."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._data_lines: list[str] = []

    def feed(self, chunk: bytes) -> list[str]:
        self._buffer.extend(chunk)
        events: list[str] = []
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                break
            raw = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            line = raw.decode("utf-8", errors="strict")
            if line == "":
                if self._data_lines:
                    events.append("\n".join(self._data_lines))
                    self._data_lines.clear()
                continue
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if field == "data":
                self._data_lines.append(value)
        return events


def semantic_delta(value: dict[str, Any]) -> bool:
    choices = value.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        for key in ("content", "reasoning_content", "tool_calls"):
            item = delta.get(key)
            if isinstance(item, str) and item:
                return True
            if isinstance(item, list) and item:
                return True
    return False


def finish_reasons(value: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    choices = value.get("choices")
    if not isinstance(choices, list):
        return reasons
    for choice in choices:
        if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str):
            reasons.append(choice["finish_reason"])
    return reasons


def parse_token_timing(
    value: Any,
    expected_index: int,
    previous_ready_ms: float | None = None,
) -> tuple[float, float] | None:
    if not isinstance(value, dict) or value.get("object") != "kiln.token_timing":
        return None
    if set(value) != {
        "object",
        "token_index",
        "ready_ms",
        "handler_received_ms",
        "queue_delay_ms",
    }:
        raise QualificationError("token timing object has an unexpected shape")
    token_index = value["token_index"]
    if (
        isinstance(token_index, bool)
        or not isinstance(token_index, int)
        or token_index != expected_index
    ):
        raise QualificationError(
            f"token timing index {token_index!r} does not match expected {expected_index}"
        )
    numbers: dict[str, float] = {}
    for field in ("ready_ms", "handler_received_ms", "queue_delay_ms"):
        raw = value[field]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise QualificationError(f"token timing {field} is not numeric")
        number = float(raw)
        if not math.isfinite(number) or number < 0:
            raise QualificationError(f"token timing {field} is not finite and nonnegative")
        numbers[field] = number
    if numbers["handler_received_ms"] + 1e-6 < numbers["ready_ms"]:
        raise QualificationError("token timing handler timestamp precedes ready timestamp")
    if previous_ready_ms is not None and numbers["ready_ms"] < previous_ready_ms:
        raise QualificationError(
            f"token timing ready_ms regressed from {previous_ready_ms} "
            f"to {numbers['ready_ms']}"
        )
    expected_delay = numbers["handler_received_ms"] - numbers["ready_ms"]
    if abs(numbers["queue_delay_ms"] - expected_delay) > 0.05:
        raise QualificationError("token timing queue delay is internally inconsistent")
    return numbers["ready_ms"], numbers["queue_delay_ms"]


def token_timing_matches_usage(
    finish_reason: str | None, timing_count: int, completion_tokens: int
) -> bool:
    """Match batching timing events to usage, accounting only for consumed EOS.

    Length and stop-sequence completions report one timing event per usage token.
    EOS maps to the same public ``stop`` reason, but the engine consumes EOS before
    emitting an EngineEvent and usage intentionally includes that one token.
    """
    if timing_count <= 0 or completion_tokens <= 0:
        return False
    if finish_reason == "length":
        return timing_count == completion_tokens
    if finish_reason == "stop":
        return completion_tokens in {timing_count, timing_count + 1}
    return False


@dataclasses.dataclass
class StreamResult:
    name: str
    marker: str
    started: float
    finished: float
    semantic_times: list[float]
    token_ready_times: list[float]
    token_queue_delays_ms: list[float]
    prompt_tokens: int
    completion_tokens: int
    usage_records: int
    finish_reason: str | None
    done: bool
    cancelled: bool
    error: str | None

    @property
    def success(self) -> bool:
        return (
            self.error is None
            and not self.cancelled
            and self.done
            and self.finish_reason in {"length", "stop"}
            and self.usage_records == 1
            and self.prompt_tokens > 0
            and self.completion_tokens > 0
            and token_timing_matches_usage(
                self.finish_reason,
                len(self.token_ready_times),
                self.completion_tokens,
            )
            and bool(self.semantic_times)
        )

    @property
    def ttft_ms(self) -> float:
        if not self.semantic_times:
            return 0.0
        return (self.semantic_times[0] - self.started) * 1000.0

    @property
    def e2e_ms(self) -> float:
        return (self.finished - self.started) * 1000.0

    @property
    def itl_ms(self) -> list[float]:
        return [
            (after - before) * 1000.0
            for before, after in zip(self.token_ready_times, self.token_ready_times[1:])
        ]


def request_body(prompt: str, max_tokens: int, seed: int) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "adapter": None,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "include_performance": True,
    }


def deterministic_prompt(marker: str, words: int) -> str:
    payload = " ".join(f"item{index % 97:02d}" for index in range(words))
    return (
        f"{marker} Read the deterministic sequence and then answer with concise numbered facts. "
        f"Do not repeat the sequence. Sequence: {payload}"
    )


def slow_consumer_prompt(marker: str) -> str:
    return (
        f"{marker} Emit one continuous plain-text sequence of ascending zero-padded integers, "
        "starting at 000000 and separated only by spaces. Continue without commentary, a "
        "summary, or an early stop until the response token limit terminates generation."
    )


def connect_slow_consumer_socket(
    port: int, socket_factory: Any | None = None
) -> socket.socket:
    factory = socket.socket if socket_factory is None else socket_factory
    sock = factory(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(15.0)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SLOW_SOCKET_BUFFER_BYTES)
        tcp_window_clamp = getattr(socket, "TCP_WINDOW_CLAMP", None)
        if tcp_window_clamp is None:
            raise QualificationError(
                "slow-consumer qualification requires TCP_WINDOW_CLAMP support"
            )
        sock.setsockopt(
            socket.IPPROTO_TCP, tcp_window_clamp, SLOW_SOCKET_BUFFER_BYTES
        )
        sock.connect(("127.0.0.1", port))
        return sock
    except Exception:
        sock.close()
        raise


def json_request(port: int, method: str, path: str, body: Any | None = None) -> Any:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    try:
        payload = None if body is None else json.dumps(body, separators=(",", ":"))
        headers = {"Accept": "application/json", "User-Agent": "kiln-qualification/1"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise QualificationError(f"{method} {path} returned HTTP {response.status}: {raw[:512]!r}")
        return json.loads(raw)
    finally:
        connection.close()


def text_request(port: int, path: str) -> str:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    try:
        connection.request("GET", path, headers={"User-Agent": "kiln-qualification/1"})
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise QualificationError(f"GET {path} returned HTTP {response.status}")
        return raw.decode("utf-8")
    finally:
        connection.close()


def read_stream_chunk(
    connection: http.client.HTTPConnection,
    response: http.client.HTTPResponse,
    *,
    deadline: float,
    abort_event: threading.Event | None,
    name: str,
) -> bytes:
    """Wait for stream data without timing out HTTPResponse's buffered reader."""
    sock = connection.sock
    if sock is None:
        raise ConnectionError(f"{name} HTTP connection has no live socket")
    while True:
        if abort_event is not None and abort_event.is_set():
            raise QualificationError(f"{name} aborted by qualification cleanup")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{name} exceeded its request or overall deadline")

        # getresponse() parses headers through a BufferedReader and may already
        # have pulled body bytes out of the kernel socket. Probe that buffer in
        # nonblocking mode before polling the socket itself, otherwise a full
        # SSE response buffered with the headers can be starved on keep-alive.
        buffered = b""
        sock.setblocking(False)
        try:
            if response.fp is not None:
                buffered = response.fp.peek(1)
        except (BlockingIOError, InterruptedError, socket.timeout):
            buffered = b""
        finally:
            # Any timeout from the actual buffered read is terminal at the
            # request deadline; it is never retried on the same HTTPResponse.
            sock.settimeout(remaining)
        if buffered:
            return response.read1(4096)

        try:
            readable, _, exceptional = select.select(
                [sock], [], [sock], min(remaining, STREAM_READ_POLL_SECONDS)
            )
        except InterruptedError:
            continue
        if exceptional:
            raise ConnectionError(f"{name} stream socket reported an exceptional condition")
        if readable:
            sock.settimeout(max(0.1, deadline - time.monotonic()))
            return response.read1(4096)


def run_stream(
    port: int,
    *,
    name: str,
    marker: str,
    prompt_words: int,
    max_tokens: int,
    seed: int,
    first_token_event: threading.Event | None = None,
    cancel_after: int | None = None,
    absolute_deadline: float | None = None,
    abort_event: threading.Event | None = None,
) -> StreamResult:
    started = time.monotonic()
    deadline = started + REQUEST_TIMEOUT_SECONDS
    if absolute_deadline is not None:
        deadline = min(deadline, absolute_deadline)
    semantic_times: list[float] = []
    token_ready_times: list[float] = []
    token_queue_delays_ms: list[float] = []
    previous_ready_ms: float | None = None
    prompt_tokens = 0
    completion_tokens = 0
    usage_records = 0
    reasons: list[str] = []
    done = False
    cancelled = False
    error: str | None = None
    connection = http.client.HTTPConnection(
        "127.0.0.1", port, timeout=max(0.1, deadline - time.monotonic())
    )
    try:
        if abort_event is not None and abort_event.is_set():
            raise QualificationError(f"{name} aborted before dispatch")
        body = request_body(
            deterministic_prompt(marker, prompt_words),
            max_tokens,
            seed,
        )
        payload = json.dumps(body, separators=(",", ":"))
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "User-Agent": "kiln-qualification/1",
                "X-Kiln-Client": f"qualification-{marker}",
            },
        )
        response = connection.getresponse()
        content_type = response.getheader("Content-Type", "")
        if response.status != 200:
            raise QualificationError(f"{name} returned HTTP {response.status}: {response.read(512)!r}")
        if "text/event-stream" not in content_type.lower():
            raise QualificationError(f"{name} returned unexpected content type {content_type!r}")
        if connection.sock is None:
            raise ConnectionError(f"{name} HTTP connection has no live socket")
        parser = SSEParser()
        while not done:
            chunk = read_stream_chunk(
                connection,
                response,
                deadline=deadline,
                abort_event=abort_event,
                name=name,
            )
            if not chunk:
                break
            observed = time.monotonic()
            for data in parser.feed(chunk):
                if data == "[DONE]":
                    done = True
                    break
                value = json.loads(data)
                timing = parse_token_timing(
                    value,
                    len(token_ready_times) + 1,
                    previous_ready_ms,
                )
                if timing is not None:
                    ready_ms, queue_delay_ms = timing
                    previous_ready_ms = ready_ms
                    token_ready_times.append(started + ready_ms / 1000.0)
                    token_queue_delays_ms.append(queue_delay_ms)
                    continue
                if semantic_delta(value):
                    semantic_times.append(observed)
                    if first_token_event is not None:
                        first_token_event.set()
                    if cancel_after is not None and len(semantic_times) >= cancel_after:
                        cancelled = True
                        if connection.sock is not None:
                            connection.sock.setsockopt(
                                socket.SOL_SOCKET,
                                socket.SO_LINGER,
                                struct.pack("ii", 1, 0),
                            )
                        return StreamResult(
                            name,
                            marker,
                            started,
                            time.monotonic(),
                            semantic_times,
                            token_ready_times,
                            token_queue_delays_ms,
                            0,
                            0,
                            0,
                            None,
                            False,
                            True,
                            None,
                        )
                reasons.extend(finish_reasons(value))
                if "usage" in value:
                    usage = value["usage"]
                    if not isinstance(usage, dict):
                        raise QualificationError(f"{name} emitted malformed usage")
                    usage_records += 1
                    if usage_records != 1:
                        raise QualificationError(f"{name} emitted multiple usage records")
                    parsed_usage: dict[str, int] = {}
                    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        token_count = usage.get(field)
                        if (
                            isinstance(token_count, bool)
                            or not isinstance(token_count, int)
                            or token_count < 0
                        ):
                            raise QualificationError(
                                f"{name} usage.{field} must be a nonnegative integer"
                            )
                        parsed_usage[field] = token_count
                    if parsed_usage["total_tokens"] != (
                        parsed_usage["prompt_tokens"]
                        + parsed_usage["completion_tokens"]
                    ):
                        raise QualificationError(f"{name} usage token totals are inconsistent")
                    prompt_tokens = parsed_usage["prompt_tokens"]
                    completion_tokens = parsed_usage["completion_tokens"]
        if len(reasons) != 1:
            raise QualificationError(f"{name} emitted {len(reasons)} finish reasons")
        if not done:
            raise QualificationError(f"{name} stream ended without [DONE]")
        if usage_records != 1 or prompt_tokens <= 0 or completion_tokens <= 0:
            raise QualificationError(f"{name} did not emit one positive token-usage record")
        if not token_timing_matches_usage(
            reasons[0], len(token_ready_times), completion_tokens
        ):
            raise QualificationError(
                f"{name} emitted {len(token_ready_times)} token timings for "
                f"{completion_tokens} completion usage tokens with finish_reason={reasons[0]!r}; "
                "length requires an exact match and stop permits only one additional "
                "non-emitted EOS usage token"
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        connection.close()
    return StreamResult(
        name=name,
        marker=marker,
        started=started,
        finished=time.monotonic(),
        semantic_times=semantic_times,
        token_ready_times=token_ready_times,
        token_queue_delays_ms=token_queue_delays_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        usage_records=usage_records,
        finish_reason=reasons[0] if len(reasons) == 1 else None,
        done=done,
        cancelled=cancelled,
        error=error,
    )


class SlowConsumer:
    def __init__(self, port: int, marker: str, seed: int) -> None:
        self.port = port
        self.marker = marker
        self.seed = seed
        self.header_received = threading.Event()
        self.stop = threading.Event()
        self.started = 0.0
        self.header_time = 0.0
        self.closed_time = 0.0
        self.status: int | None = None
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, name="qualification-slow-client")
        self._started = False

    def start(self) -> None:
        self.thread.start()
        self._started = True

    def close(self) -> None:
        self.stop.set()
        if not self._started:
            return
        self.thread.join(timeout=10.0)
        if self.thread.is_alive() and self.error is None:
            self.error = "slow-consumer thread did not stop within 10 seconds"

    def _run(self) -> None:
        sock: socket.socket | None = None
        self.started = time.monotonic()
        try:
            sock = connect_slow_consumer_socket(self.port)
            body = json.dumps(
                request_body(
                    slow_consumer_prompt(self.marker),
                    SLOW_MAX_TOKENS,
                    self.seed,
                ),
                separators=(",", ":"),
            ).encode("utf-8")
            request = (
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                + f"Host: 127.0.0.1:{self.port}\r\n".encode()
                + b"Accept: text/event-stream\r\n"
                + b"Content-Type: application/json\r\n"
                + b"User-Agent: kiln-qualification-slow/1\r\n"
                + f"X-Kiln-Client: qualification-{self.marker}\r\n".encode()
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + body
            )
            sock.sendall(request)
            header = bytearray()
            deadline = time.monotonic() + 30.0
            while b"\r\n\r\n" not in header:
                if time.monotonic() >= deadline:
                    raise TimeoutError("slow consumer did not receive response headers")
                chunk = sock.recv(1)
                if not chunk:
                    raise QualificationError("slow consumer connection closed before headers")
                header.extend(chunk)
                if len(header) > 64 * 1024:
                    raise QualificationError("slow consumer response headers exceeded 64 KiB")
            status_line = bytes(header).split(b"\r\n", 1)[0].decode("ascii", errors="replace")
            match = re.match(r"HTTP/1\.[01] (\d{3})", status_line)
            if match is None:
                raise QualificationError(f"malformed slow consumer status: {status_line!r}")
            self.status = int(match.group(1))
            if self.status != 200:
                raise QualificationError(f"slow consumer returned HTTP {self.status}")
            self.header_time = time.monotonic()
            self.header_received.set()
            self.stop.wait(OVERALL_TIMEOUT_SECONDS)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.header_received.set()
        finally:
            if sock is not None:
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                except OSError:
                    pass
                sock.close()
            self.closed_time = time.monotonic()


def parse_prometheus_used_bytes(text: str) -> int | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("kiln_gpu_memory_bytes"):
            continue
        name, separator, raw_value = line.rpartition(" ")
        if not separator or 'kind="used"' not in name:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0:
            return int(value)
    return None


class MemorySampler:
    def __init__(self, port: int) -> None:
        self.port = port
        self.stop = threading.Event()
        self.samples: list[int] = []
        self.errors: list[str] = []
        self.thread = threading.Thread(target=self._run, name="qualification-memory-sampler")
        self._started = False

    def start(self) -> None:
        self.thread.start()
        self._started = True

    def close(self) -> None:
        self.stop.set()
        if not self._started:
            return
        self.thread.join(timeout=10.0)
        if self.thread.is_alive() and len(self.errors) < 8:
            self.errors.append("memory-sampler thread did not stop within 10 seconds")

    def _run(self) -> None:
        while not self.stop.wait(MEMORY_POLL_INTERVAL_SECONDS):
            try:
                value = parse_prometheus_used_bytes(text_request(self.port, "/metrics"))
                if value is not None:
                    self.samples.append(value)
            except Exception as exc:
                if len(self.errors) < 8:
                    self.errors.append(f"{type(exc).__name__}: {exc}")


@dataclasses.dataclass(frozen=True)
class ObservedEvent:
    observed: float
    category: str
    message: str
    fields: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class DeliveryPressureWindow:
    request_id: str
    client: str
    started: float
    timed_out: float


def parse_server_log_line(line: str) -> tuple[str, dict[str, Any]]:
    message = line
    structured_fields: dict[str, Any] = {}
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return message, structured_fields
    if not isinstance(value, dict):
        return message, structured_fields
    fields = value.get("fields")
    if isinstance(fields, dict):
        structured_fields = dict(fields)
        if isinstance(fields.get("message"), str):
            message = fields["message"]
    elif isinstance(value.get("message"), str):
        message = value["message"]
    return message, structured_fields


def classify_server_event(
    message: str, fields: dict[str, Any] | None = None
) -> str | None:
    event_name = fields.get("event") if isinstance(fields, dict) else None
    lowered = (event_name if isinstance(event_name, str) else message).strip().lower()
    if lowered == "background inference prewarm complete":
        return "prewarm_complete"
    if lowered == "kv autoscaler resized cache":
        return "kv_resize"
    if lowered in {
        "memory governor: reclaimed under pressure",
        "rocm pool reclaim completed",
    }:
        return "memory_reclaim"
    if lowered.startswith("rocm graph capture failed:") or lowered.startswith(
        "rocm graph replay failed:"
    ):
        return "graph_fallback"
    if lowered.startswith("rocm hip graph captured for decode"):
        return "graph_capture"
    if lowered == "stream_request_bound":
        return "stream_request_bound"
    if lowered == "response_channel_backpressure":
        return "client_backpressure_start"
    if lowered == "response_channel_backpressure_timeout":
        return "client_backpressure_timeout"
    return None


def attributed_delivery_pressure_window(
    events: list[ObservedEvent], expected_client: str
) -> DeliveryPressureWindow | None:
    bindings = [
        event
        for event in events
        if event.category == "stream_request_bound"
        and event.fields.get("client") == expected_client
    ]
    request_ids = {
        event.fields.get("request_id")
        for event in bindings
        if isinstance(event.fields.get("request_id"), str)
        and event.fields.get("request_id")
    }
    if len(request_ids) > 1:
        raise QualificationError(
            f"slow-consumer marker bound to multiple request IDs: {sorted(request_ids)}"
        )
    if not request_ids:
        return None
    request_id = next(iter(request_ids))
    binding_time = min(
        event.observed
        for event in bindings
        if event.fields.get("request_id") == request_id
    )
    starts = [
        event
        for event in events
        if event.category == "client_backpressure_start"
        and event.fields.get("request_id") == request_id
        and event.observed >= binding_time
    ]
    timeouts = [
        event
        for event in events
        if event.category == "client_backpressure_timeout"
        and event.fields.get("request_id") == request_id
        and event.observed >= binding_time
    ]
    if len(starts) > 1 or len(timeouts) > 1:
        raise QualificationError(
            "slow-consumer request emitted duplicate backpressure start or timeout evidence"
        )
    if timeouts and not starts:
        raise QualificationError(
            "slow-consumer request emitted a backpressure timeout without a start"
        )
    if not starts or not timeouts:
        return None
    if timeouts[0].observed < starts[0].observed:
        raise QualificationError(
            "slow-consumer backpressure timeout preceded its start"
        )
    return DeliveryPressureWindow(
        request_id=request_id,
        client=expected_client,
        started=starts[0].observed,
        timed_out=timeouts[0].observed,
    )


def healthy_peer_overlaps_pressure(
    result: StreamResult, pressure: DeliveryPressureWindow | None
) -> bool:
    if pressure is None:
        return False
    return (
        result.success
        and result.started <= pressure.timed_out
        and result.finished >= pressure.started
        and any(ready < pressure.started for ready in result.token_ready_times)
        and any(
            pressure.started <= ready <= pressure.timed_out
            for ready in result.token_ready_times
        )
        and any(ready > pressure.timed_out for ready in result.token_ready_times)
    )


class ServerLog:
    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.events: list[ObservedEvent] = []
        self._events_lock = threading.Lock()
        self.tail: deque[str] = deque(maxlen=200)
        self.prewarm_complete = threading.Event()
        self._stderr_bytes = 0
        self.thread = threading.Thread(target=self._run, name="qualification-server-log")

    def start(self) -> None:
        self.thread.start()

    def join(self) -> None:
        self.thread.join(timeout=10.0)

    def events_since(self, started: float) -> list[ObservedEvent]:
        with self._events_lock:
            return [event for event in self.events if event.observed >= started]

    def _run(self) -> None:
        for raw in self.stream:
            line = raw.rstrip("\n")
            self.tail.append(line)
            if self._stderr_bytes < 4 * 1024 * 1024:
                encoded = (line + "\n").encode("utf-8", errors="replace")
                remaining = 4 * 1024 * 1024 - self._stderr_bytes
                sys.stderr.buffer.write(encoded[:remaining])
                sys.stderr.buffer.flush()
                self._stderr_bytes += min(len(encoded), remaining)
            message = line
            message, fields = parse_server_log_line(line)
            category = classify_server_event(message, fields)
            if category is not None:
                event = ObservedEvent(
                    time.monotonic(), category, message[:512], fields
                )
                with self._events_lock:
                    self.events.append(event)
                if category == "prewarm_complete":
                    self.prewarm_complete.set()
                trace("server_event", category=category, message=message[:512])


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sanitized_environment(source: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in source.items()
        if not key.startswith("KILN_") and key not in {"RUST_LOG", "KILN_CONFIG"}
    }


def source_bound_build_command() -> list[str]:
    return [
        "cargo",
        "build",
        "--quiet",
        f"--{BUILD_PROFILE}",
        "--locked",
        "--offline",
        "-p",
        BUILD_PACKAGE,
        "--bin",
        BUILD_BINARY,
        "--no-default-features",
        "--features",
        BUILD_FEATURES,
    ]


def build_binary(absolute_deadline: float) -> tuple[Path, str, float]:
    started = time.monotonic()
    environment = sanitized_environment(dict(os.environ))
    environment.update(
        {
            "CARGO_NET_OFFLINE": "true",
            "KILN_ROCM_ARCHS": BUILD_ROCM_ARCHS,
            "ROCM_PATH": BUILD_ROCM_PATH,
        }
    )
    command = source_bound_build_command()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=remaining_until(
            absolute_deadline, "source-bound ROCm build", STARTUP_TIMEOUT_SECONDS
        ),
        check=False,
    )
    if completed.returncode != 0:
        tail = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise QualificationError(f"source-bound ROCm build failed ({completed.returncode}): {tail}")
    binary = ROOT / "target" / BUILD_PROFILE / BUILD_BINARY
    if not binary.is_file():
        raise QualificationError(f"build succeeded without {binary}")
    return binary, sha256_file(binary), time.monotonic() - started


def server_environment(
    variant: str, model_path: Path, port: int, adapter_dir: Path
) -> dict[str, str]:
    config = VARIANT_CONFIGS[variant]
    environment = sanitized_environment(dict(os.environ))
    environment.update(
        {
            "KILN_ADAPTER_DIR": str(adapter_dir),
            "KILN_CHAT_PERFORMANCE_METADATA": (
                "true" if config["server"]["chat_performance_metadata_enabled"] else "false"
            ),
            "KILN_DEBUG_ENDPOINTS": (
                "1" if config["server"]["debug_endpoints_enabled"] else "0"
            ),
            "KILN_DEFAULT_THINKING_ENABLED": "false",
            "KILN_HOST": "127.0.0.1",
            "KILN_HTTP_SEND_BUFFER_BYTES": str(
                config["server"]["http_send_buffer_bytes"]
            ),
            "KILN_LOG_FORMAT": config["server"]["log_format"],
            "KILN_MEMORY_RECLAIM_MODE": config["runtime"]["memory_reclaim_mode"],
            "KILN_MODEL_PATH": str(model_path),
            "KILN_PORT": str(port),
            "KILN_REQUEST_TIMEOUT_SECS": str(
                config["server"]["request_timeout_seconds"]
            ),
            "KILN_SERVED_MODEL_ID": MODEL_ID,
            "KILN_STREAM_STALL_GRACE_MS": str(
                config["server"]["stream_stall_grace_ms"]
            ),
            "RUST_LOG": "kiln=info,kiln_server=info,kiln_model=info,kiln_memory=info,tower_http=warn",
        }
    )
    if not config["runtime"]["kv_autoscale_enabled"]:
        environment["KILN_KV_AUTOSCALE"] = "0"
    if not config["runtime"]["rocm_graphs_enabled"]:
        environment["KILN_ROCM_GRAPHS"] = "0"
    return environment


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10.0)


def wait_ready(
    port: int,
    process: subprocess.Popen[str],
    server_log: ServerLog,
    absolute_deadline: float,
) -> dict[str, Any]:
    deadline = min(
        time.monotonic() + STARTUP_TIMEOUT_SECONDS,
        absolute_deadline,
    )
    last_error = "server not queried"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = "\n".join(server_log.tail)
            raise QualificationError(f"server exited during startup ({process.returncode}):\n{tail}")
        try:
            health = json_request(port, "GET", "/health")
            checks = health.get("checks")
            if (
                health.get("status") == "ok"
                and isinstance(checks, list)
                and checks
                and all(item.get("pass") is True for item in checks if isinstance(item, dict))
                and any(
                    item.get("name") == "inference_prewarm_complete"
                    and item.get("pass") is True
                    for item in checks
                    if isinstance(item, dict)
                )
            ):
                if not server_log.prewarm_complete.wait(
                    timeout=remaining_until(deadline, "server prewarm log", 5.0)
                ):
                    raise QualificationError("health passed without prewarm completion log evidence")
                return health
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1.0)
    raise QualificationError(f"server readiness exceeded {STARTUP_TIMEOUT_SECONDS:g}s: {last_error}")


def http_send_buffer_attestation_failures(
    value: Any,
    *,
    label: str,
    expected_requested_bytes: int,
    platform_name: str | None = None,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} HTTP send-buffer runtime state is missing"]
    failures: list[str] = []

    requested = value.get("send_buffer_requested_bytes")
    if (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested != expected_requested_bytes
    ):
        failures.append(
            f"{label} HTTP send-buffer requested bytes={requested!r}, "
            f"expected {expected_requested_bytes}"
        )

    raw = value.get("send_buffer_kernel_readback_bytes")
    raw_valid = not isinstance(raw, bool) and isinstance(raw, int) and raw > 0
    if not raw_valid:
        failures.append(
            f"{label} HTTP send-buffer kernel read-back must be a positive integer, "
            f"got {raw!r}"
        )

    effective = value.get("send_buffer_effective_bytes")
    effective_valid = (
        not isinstance(effective, bool) and isinstance(effective, int) and effective > 0
    )
    if not effective_valid:
        failures.append(
            f"{label} HTTP send-buffer effective bytes must be a positive integer, "
            f"got {effective!r}"
        )
    elif effective < expected_requested_bytes:
        failures.append(
            f"{label} HTTP send-buffer effective bytes={effective}, below requested "
            f"{expected_requested_bytes}"
        )

    if raw_valid and effective_valid:
        platform = sys.platform if platform_name is None else platform_name
        expected_raw = effective * 2 if platform.startswith("linux") else effective
        if raw != expected_raw:
            relationship = "twice effective bytes" if platform.startswith("linux") else "effective bytes"
            failures.append(
                f"{label} HTTP send-buffer kernel read-back={raw} must equal "
                f"{relationship} ({expected_raw}) on {platform}"
            )
    return failures


def gpu_memory_attestation_failures(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["health.gpu_memory is missing"]
    failures: list[str] = []
    total_vram = value.get("total_vram_bytes")
    if (
        isinstance(total_vram, bool)
        or not isinstance(total_vram, int)
        or total_vram <= 0
    ):
        failures.append(
            "health.gpu_memory.total_vram_bytes must be a positive integer"
        )
    live = value.get("live")
    if not isinstance(live, dict):
        failures.append("health.gpu_memory.live is missing")
        return failures
    used_gb = live.get("used_gb")
    if (
        isinstance(used_gb, bool)
        or not isinstance(used_gb, (int, float))
        or not math.isfinite(float(used_gb))
        or used_gb < 0
    ):
        failures.append(
            "health.gpu_memory.live.used_gb must be finite and nonnegative"
        )
    source = live.get("source")
    if not isinstance(source, str) or not source.strip():
        failures.append("health.gpu_memory.live.source must be a nonempty string")
    return failures


def attest_runtime(
    variant: str, health: dict[str, Any], debug: dict[str, Any]
) -> list[str]:
    expected = VARIANT_CONFIGS[variant]["runtime"]
    failures: list[str] = []
    if health.get("backend") != "model":
        failures.append(f"health.backend={health.get('backend')!r}, expected 'model'")
    failures.extend(gpu_memory_attestation_failures(health.get("gpu_memory")))
    expected_send_buffer = VARIANT_CONFIGS[variant]["server"]["http_send_buffer_bytes"]
    health_http = health.get("http")
    debug_http = debug.get("http")
    failures.extend(
        http_send_buffer_attestation_failures(
            health_http,
            label="health",
            expected_requested_bytes=expected_send_buffer,
        )
    )
    failures.extend(
        http_send_buffer_attestation_failures(
            debug_http,
            label="debug",
            expected_requested_bytes=expected_send_buffer,
        )
    )
    if (
        isinstance(health_http, dict)
        and isinstance(debug_http, dict)
        and health_http != debug_http
    ):
        failures.append("health and debug HTTP send-buffer runtime state disagree exactly")
    runtime = health.get("decode_runtime")
    if not isinstance(runtime, dict):
        return failures + ["health.decode_runtime is missing"]
    graph = runtime.get("rocm_graphs")
    expected_graphs = expected["rocm_graphs_enabled"]
    if not isinstance(graph, dict):
        failures.append("ROCm graph runtime state is missing")
    else:
        for field in ("requested", "capture_requested", "enabled", "capture_enabled"):
            if graph.get(field) is not expected_graphs:
                failures.append(
                    f"ROCm graph {field}={graph.get(field)!r}, expected {expected_graphs}"
                )
        expected_state = "enabled" if expected_graphs else "disabled"
        if graph.get("state") != expected_state:
            failures.append(
                f"ROCm graph state={graph.get('state')!r}, expected {expected_state!r}"
            )
    autoscaler = runtime.get("kv_autoscaler")
    expected_autoscaler = expected["kv_autoscale_enabled"]
    if not isinstance(autoscaler, dict):
        failures.append("KV autoscaler runtime state is missing")
    else:
        expected_autoscaler_fields = {
            "requested": expected_autoscaler,
            "enabled": expected_autoscaler,
            "state": "enabled" if expected_autoscaler else "disabled",
            "reason": "active" if expected_autoscaler else "environment",
        }
        for field, value in expected_autoscaler_fields.items():
            if autoscaler.get(field) != value:
                failures.append(
                    f"KV autoscaler {field}={autoscaler.get(field)!r}, expected {value!r}"
                )
    governor = runtime.get("memory_governor")
    if not isinstance(governor, dict) or governor.get("reclaim_mode") != expected[
        "memory_reclaim_mode"
    ]:
        failures.append(f"memory reclaim mode does not match {expected['memory_reclaim_mode']!r}")
    elif governor.get("automatic_monitor_enabled") is not False:
        failures.append("memory governor automatic monitor unexpectedly enabled")
    elif governor.get("source") != "environment":
        failures.append("memory reclaim mode was not sourced from the isolated environment")
    batching = runtime.get("batching_engine")
    if not isinstance(batching, dict):
        failures.append("batching engine is not enabled")
    else:
        expected_stall_grace = VARIANT_CONFIGS[variant]["server"][
            "stream_stall_grace_ms"
        ]
        if batching.get("stream_stall_grace_ms") != expected_stall_grace:
            failures.append("health batching stream-stall grace does not match config")
        if batching.get("stream_stall_grace_source") != "environment":
            failures.append("health batching stream-stall grace source is not environment")

        debug_batching = debug.get("batching_engine")
        debug_snapshot = (
            debug_batching.get("snapshot")
            if isinstance(debug_batching, dict)
            else None
        )
        if not isinstance(debug_snapshot, dict):
            failures.append("debug batching-engine snapshot is missing")
        elif (
            debug_snapshot.get("stream_stall_grace_ms") != expected_stall_grace
            or debug_snapshot.get("stream_stall_grace_source") != "environment"
        ):
            failures.append("debug batching stream-stall policy does not match environment")

    flags = debug.get("env_flags")
    if not isinstance(flags, dict):
        failures.append("debug env_flags are missing")
        return failures
    for name, enabled in (
        ("KILN_KV_AUTOSCALE", expected["kv_autoscale_enabled"]),
        ("KILN_ROCM_GRAPHS", expected["rocm_graphs_enabled"]),
    ):
        state = flags.get(name)
        if not isinstance(state, dict):
            failures.append(f"debug flag {name} is missing")
        elif enabled and state.get("present") is not False:
            failures.append(f"default-on flag {name} must remain absent")
        elif not enabled and (state.get("present") is not True or state.get("value") != "0"):
            failures.append(f"disabled flag {name} must be present with value 0")
    memory_flag = flags.get("KILN_MEMORY_RECLAIM_MODE")
    if not isinstance(memory_flag, dict) or memory_flag.get("value") != expected[
        "memory_reclaim_mode"
    ]:
        failures.append("memory reclaim debug flag does not match effective mode")
    send_buffer_flag = flags.get("KILN_HTTP_SEND_BUFFER_BYTES")
    if (
        not isinstance(send_buffer_flag, dict)
        or send_buffer_flag.get("present") is not True
        or send_buffer_flag.get("value") != str(expected_send_buffer)
    ):
        failures.append("HTTP send-buffer debug flag does not match effective mode")
    stall_grace_flag = flags.get("KILN_STREAM_STALL_GRACE_MS")
    if (
        not isinstance(stall_grace_flag, dict)
        or stall_grace_flag.get("present") is not True
        or stall_grace_flag.get("value")
        != str(VARIANT_CONFIGS[variant]["server"]["stream_stall_grace_ms"])
    ):
        failures.append("stream-stall grace debug flag does not match effective policy")
    return failures


def batching_snapshot(health: dict[str, Any]) -> dict[str, int]:
    runtime = health.get("decode_runtime")
    batching = runtime.get("batching_engine") if isinstance(runtime, dict) else None
    scheduler = health.get("scheduler")
    if not isinstance(batching, dict):
        raise QualificationError("health batching-engine snapshot is missing")
    if not isinstance(scheduler, dict):
        raise QualificationError("health scheduler snapshot is missing")
    snapshot: dict[str, int] = {}
    for field in (
        "max_observed_batch_size",
        "total_errors",
        "total_decode_forwards",
        "total_batched_decode_forwards",
        "total_decode_rows",
        "response_backpressure_events",
        "response_backpressure_wait_ms",
        "response_stall_evictions",
        "response_channel_closed",
    ):
        value = batching.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise QualificationError(
                f"batching-engine field {field} must be a nonnegative integer, got {value!r}"
            )
        snapshot[field] = value
    for source, field in ((scheduler, "blocks_total"), (scheduler, "blocks_used")):
        value = source.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise QualificationError(
                f"scheduler field {field} must be a nonnegative integer, got {value!r}"
            )
        snapshot[field] = value
    return snapshot


def graph_snapshot(health: dict[str, Any]) -> dict[str, int]:
    runtime = health.get("decode_runtime")
    if not isinstance(runtime, dict):
        raise QualificationError("health.decode_runtime is missing")
    graph = runtime.get("rocm_graphs")
    if not isinstance(graph, dict):
        raise QualificationError("health.decode_runtime.rocm_graphs is missing")
    snapshot: dict[str, int] = {}
    for field in (*GRAPH_MONOTONIC_FIELDS, *GRAPH_GAUGE_FIELDS):
        value = graph.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise QualificationError(
                f"ROCm graph field {field} must be a nonnegative integer, got {value!r}"
            )
        snapshot[field] = value
    if snapshot["capture_attempts"] != (
        snapshot["capture_successes"]
        + snapshot["capture_deferrals"]
        + snapshot["capture_failures"]
    ):
        raise QualificationError("ROCm graph capture counters violate their outcome invariant")
    if snapshot["replay_attempts"] != (
        snapshot["replay_successes"] + snapshot["replay_failures"]
    ):
        raise QualificationError("ROCm graph replay counters violate their outcome invariant")
    if snapshot["failures"] != (
        snapshot["capture_failures"] + snapshot["replay_failures"]
    ):
        raise QualificationError("ROCm graph aggregate failure counter is inconsistent")
    return snapshot


def counter_delta(before: dict[str, int], after: dict[str, int], field: str) -> int:
    if after[field] < before[field]:
        raise QualificationError(
            f"monotonic counter {field} regressed from {before[field]} to {after[field]}"
        )
    return after[field] - before[field]


def read_stable_health(
    port: int, absolute_deadline: float, label: str
) -> dict[str, Any]:
    deadline = min(time.monotonic() + 10.0, absolute_deadline)
    last_state: Any = None
    while time.monotonic() < deadline:
        health = json_request(port, "GET", "/health")
        graph = ((health.get("decode_runtime") or {}).get("rocm_graphs") or {})
        last_state = graph.get("state") if isinstance(graph, dict) else None
        if last_state != "busy":
            graph_snapshot(health)
            batching_snapshot(health)
            return health
        time.sleep(0.05)
    raise TimeoutError(f"{label} could not obtain stable graph health; last state={last_state!r}")


def attest_runtime_execution(
    variant: str,
    health_after_warmup: dict[str, Any],
    health_end: dict[str, Any],
) -> list[str]:
    expected_graphs = VARIANT_CONFIGS[variant]["runtime"]["rocm_graphs_enabled"]
    warmup = graph_snapshot(health_after_warmup)
    after = graph_snapshot(health_end)
    failures: list[str] = []
    if expected_graphs:
        if warmup["capture_successes"] < 1:
            failures.append("graph-on warmup completed without a successful capture")
        if warmup["replay_successes"] < 1:
            failures.append("graph-on warmup completed without a successful replay")
        if warmup["failures"] != 0:
            failures.append("graph-on warmup recorded a graph failure")
        if counter_delta(warmup, after, "replay_successes") < 1:
            failures.append("measured graph-on load completed without a successful replay")
        if counter_delta(warmup, after, "capture_failures") != 0:
            failures.append("measured graph-on load recorded a capture failure")
        if counter_delta(warmup, after, "replay_failures") != 0:
            failures.append("measured graph-on load recorded a replay failure")
    else:
        for window, snapshot in (("warmup", warmup), ("final", after)):
            for field, value in snapshot.items():
                if value != 0:
                    failures.append(
                        f"graph-off {window} recorded {field}={value}, expected 0"
                    )
    return failures


def wait_for_delivery_pressure(
    port: int,
    baseline: dict[str, int],
    server_log: ServerLog,
    expected_client: str,
    observed_since: float,
    absolute_deadline: float,
) -> tuple[DeliveryPressureWindow | None, bool, dict[str, Any]]:
    deadline = min(time.monotonic() + 45.0, absolute_deadline)
    latest: dict[str, Any] = {}
    pressure: DeliveryPressureWindow | None = None
    while time.monotonic() < deadline:
        latest = json_request(port, "GET", "/health")
        snapshot = batching_snapshot(latest)
        pressure = attributed_delivery_pressure_window(
            server_log.events_since(observed_since), expected_client
        )
        backpressured = counter_delta(
            baseline, snapshot, "response_backpressure_events"
        ) >= 1
        evicted = counter_delta(baseline, snapshot, "response_stall_evictions") >= 1
        if pressure is not None and backpressured and evicted:
            return pressure, True, latest
        time.sleep(0.25)
    return pressure, False, latest


def wait_for_cancellation_and_drain(
    port: int, marker: str, absolute_deadline: float
) -> tuple[bool, dict[str, Any]]:
    deadline = min(time.monotonic() + 45.0, absolute_deadline)
    cancellation = False
    drained = False
    last_health: dict[str, Any] = {}
    while time.monotonic() < deadline:
        records = json_request(port, "GET", "/v1/stats/recent-requests")
        if isinstance(records, list):
            cancellation = cancellation_recorded(records, marker)
        last_health = json_request(port, "GET", "/health")
        runtime = last_health.get("decode_runtime")
        if not isinstance(runtime, dict):
            raise QualificationError(
                "cancellation drain health.decode_runtime is missing"
            )
        batching = runtime.get("batching_engine")
        drained = batching_engine_drained(batching)
        if cancellation and drained:
            return True, last_health
        time.sleep(0.5)
    return cancellation and drained, last_health


def batching_engine_drained(batching: Any) -> bool:
    if not isinstance(batching, dict):
        raise QualificationError("health batching-engine drain snapshot is missing")
    values: dict[str, int] = {}
    for field in ("active_decode", "queue_depth"):
        value = batching.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise QualificationError(
                f"batching-engine drain field {field} must be a nonnegative integer, "
                f"got {value!r}"
            )
        values[field] = value
    return values["active_decode"] == 0 and values["queue_depth"] == 0


def cancellation_recorded(records: list[Any], marker: str) -> bool:
    return any(
        isinstance(record, dict)
        and marker in str(record.get("prompt_preview", ""))
        and record.get("finish_reason") == "client_disconnect"
        for record in records
    )


def disabled_policy_attestation_failures(
    variant: str,
    events: list[ObservedEvent],
    *,
    initial_blocks_total: int,
    final_blocks_total: int,
) -> list[str]:
    runtime = VARIANT_CONFIGS[variant]["runtime"]
    categories = [event.category for event in events]
    failures: list[str] = []
    if (
        runtime["memory_reclaim_mode"] == "off"
        and "memory_reclaim" in categories
    ):
        failures.append(
            "memory-reclaim-off policy observed a reclaim event during startup or load"
        )
    if not runtime["kv_autoscale_enabled"]:
        if "kv_resize" in categories:
            failures.append(
                "KV-autoscale-off policy observed a resize event during startup or load"
            )
        if final_blocks_total != initial_blocks_total:
            failures.append(
                "KV-autoscale-off policy changed blocks_total from "
                f"{initial_blocks_total} to {final_blocks_total}"
            )
    if not runtime["rocm_graphs_enabled"] and any(
        category in {"graph_capture", "graph_fallback"} for category in categories
    ):
        failures.append(
            "ROCm-graphs-off policy observed graph capture or fallback during startup or load"
        )
    return failures


def classify_itl_outliers(
    warmup_gaps: list[float],
    results: list[StreamResult],
    events: list[ObservedEvent],
) -> tuple[int, int]:
    history: deque[float] = deque(
        (gap for gap in warmup_gaps if gap >= 0), maxlen=OUTLIER_HISTORY_SIZE
    )
    gaps: list[tuple[float, float, float]] = []
    for result in results:
        for before, after in zip(
            result.token_ready_times, result.token_ready_times[1:]
        ):
            gaps.append((after, before, (after - before) * 1000.0))
    gaps.sort(key=lambda item: item[0])
    attributed = 0
    unexplained = 0
    attributable = {
        "kv_resize",
        "memory_reclaim",
        "graph_capture",
        "graph_fallback",
        "client_backpressure_start",
        "client_backpressure_timeout",
    }
    for after, before, gap_ms in gaps:
        baseline = percentile_r7(history, 0.5) if history else gap_ms
        threshold = max(OUTLIER_ABSOLUTE_MS, OUTLIER_MULTIPLIER * baseline)
        if gap_ms > threshold:
            nearby = [
                event
                for event in events
                if event.category in attributable and before - 0.05 <= event.observed <= after + 0.10
            ]
            if nearby:
                attributed += 1
            else:
                unexplained += 1
            trace(
                "itl_outlier",
                attributed=bool(nearby),
                gap_ms=gap_ms,
                nearby_categories=sorted({event.category for event in nearby}),
                threshold_ms=threshold,
            )
        history.append(gap_ms)
    return attributed, unexplained


def metric_values(
    *,
    measured: list[StreamResult],
    warmup: StreamResult,
    long_prefill: StreamResult,
    cancellation_confirmed: bool,
    slow_peer_success: int,
    peak_memory: int,
    health_after_warmup: dict[str, Any],
    health_measurement_start: dict[str, Any],
    health_end: dict[str, Any],
    events: list[ObservedEvent],
) -> dict[str, float | int]:
    successes = [result for result in measured if result.success]
    ttfts = [result.ttft_ms for result in successes]
    e2es = [result.e2e_ms for result in successes]
    itls = [gap for result in successes for gap in result.itl_ms]
    queue_delays = [
        delay for result in successes for delay in result.token_queue_delays_ms
    ]
    start = min((result.started for result in measured), default=time.monotonic())
    finish = max((result.finished for result in measured), default=start)
    window = max(finish - start, 1e-9)
    completion_tokens = sum(result.completion_tokens for result in successes)
    prompt_tokens = sum(result.prompt_tokens for result in successes)
    failures = len(measured) - len(successes)
    zero_tokens = sum(result.completion_tokens == 0 for result in measured)
    slo_good = sum(
        result.ttft_ms <= SLO_TTFT_MS and result.e2e_ms <= SLO_E2E_MS
        for result in successes
    )
    attributed, unexplained = classify_itl_outliers(warmup.itl_ms, successes, events)
    batching_start = batching_snapshot(health_measurement_start)
    batching_end = batching_snapshot(health_end)
    graph_start = graph_snapshot(health_after_warmup)
    graph_end = graph_snapshot(health_end)
    decode_forwards = counter_delta(
        batching_start, batching_end, "total_decode_forwards"
    )
    batched_decode_forwards = counter_delta(
        batching_start, batching_end, "total_batched_decode_forwards"
    )
    decode_rows = counter_delta(batching_start, batching_end, "total_decode_rows")
    categories = [event.category for event in events]
    return {
        "attributed_itl_outlier_count": attributed,
        "batching_batched_decode_forward_count": batched_decode_forwards,
        "batching_decode_forward_count": decode_forwards,
        "batching_decode_row_count": decode_rows,
        "batching_max_observed_batch_size": max(
            batching_start["max_observed_batch_size"],
            batching_end["max_observed_batch_size"],
        ),
        "batching_mean_rows_per_forward": decode_rows / max(decode_forwards, 1),
        "batching_total_errors": counter_delta(
            batching_start, batching_end, "total_errors"
        ),
        "cancellation_confirmed_count": int(cancellation_confirmed),
        "client_backpressure_event_count": counter_delta(
            batching_start, batching_end, "response_backpressure_events"
        ),
        "client_backpressure_wait_ms": counter_delta(
            batching_start, batching_end, "response_backpressure_wait_ms"
        ),
        "client_stall_eviction_count": counter_delta(
            batching_start, batching_end, "response_stall_evictions"
        ),
        "completion_token_count": completion_tokens,
        "e2e_latency_ms_p50": percentile_r7(e2es, 0.5),
        "e2e_latency_ms_p99": percentile_r7(e2es, 0.99),
        "e2e_latency_ms_p999": percentile_r7(e2es, 0.999),
        "graph_measured_capture_attempt_count": counter_delta(
            graph_start, graph_end, "capture_attempts"
        ),
        "graph_measured_capture_deferral_count": counter_delta(
            graph_start, graph_end, "capture_deferrals"
        ),
        "graph_measured_capture_failure_count": counter_delta(
            graph_start, graph_end, "capture_failures"
        ),
        "graph_measured_capture_success_count": counter_delta(
            graph_start, graph_end, "capture_successes"
        ),
        "graph_measured_live_count_end": graph_end["captured_graph_count"],
        "graph_measured_replay_attempt_count": counter_delta(
            graph_start, graph_end, "replay_attempts"
        ),
        "graph_measured_replay_failure_count": counter_delta(
            graph_start, graph_end, "replay_failures"
        ),
        "graph_measured_replay_success_count": counter_delta(
            graph_start, graph_end, "replay_successes"
        ),
        "graph_pre_measurement_capture_success_count": graph_start["capture_successes"],
        "graph_pre_measurement_failure_count": graph_start["failures"],
        "graph_pre_measurement_replay_success_count": graph_start["replay_successes"],
        "itl_ms_p50": percentile_r7(itls, 0.5),
        "itl_ms_p99": percentile_r7(itls, 0.99),
        "itl_ms_p999": percentile_r7(itls, 0.999),
        "kv_blocks_end": batching_end["blocks_total"],
        "kv_blocks_start": batching_start["blocks_total"],
        "kv_resize_event_count": categories.count("kv_resize"),
        "long_prefill_prompt_tokens": long_prefill.prompt_tokens,
        "memory_reclaim_event_count": categories.count("memory_reclaim"),
        "output_token_throughput_per_second": completion_tokens / window,
        "peak_gpu_memory_used_bytes": peak_memory,
        "prompt_token_count": prompt_tokens,
        "request_count": len(measured),
        "request_failure_count": failures,
        "request_throughput_per_second": len(successes) / window,
        "response_queue_delay_ms_p50": percentile_r7(queue_delays, 0.5),
        "response_queue_delay_ms_p99": percentile_r7(queue_delays, 0.99),
        "response_queue_delay_ms_p999": percentile_r7(queue_delays, 0.999),
        "slo_goodput_requests_per_second": slo_good / window,
        "slow_consumer_peer_success_count": slow_peer_success,
        "ttft_ms_p50": percentile_r7(ttfts, 0.5),
        "ttft_ms_p99": percentile_r7(ttfts, 0.99),
        "ttft_ms_p999": percentile_r7(ttfts, 0.999),
        "unexplained_itl_outlier_count": unexplained,
        "zero_token_response_count": zero_tokens,
    }


def metrics_from_values(values: dict[str, float | int]) -> list[dict[str, Any]]:
    if set(values) != set(METRIC_DEFINITIONS):
        missing = sorted(set(METRIC_DEFINITIONS) - set(values))
        extra = sorted(set(values) - set(METRIC_DEFINITIONS))
        raise QualificationError(f"metric set mismatch: missing={missing}, extra={extra}")
    metrics = []
    for name in sorted(values):
        unit, aggregation, lower_is_better = METRIC_DEFINITIONS[name]
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise QualificationError(f"metric {name} is not finite numeric evidence")
        metrics.append(
            {
                "name": name,
                "value": value,
                "unit": unit,
                "aggregation": aggregation,
                "lower_is_better": lower_is_better,
            }
        )
    return metrics


def zero_metrics() -> list[dict[str, Any]]:
    values = {name: 0 for name in METRIC_DEFINITIONS}
    values["request_failure_count"] = 1
    return metrics_from_values(values)


def write_result(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def bounded_details(value: str | None) -> str | None:
    if value is None or len(value) <= 2000:
        return value
    return value[:1976] + "...[details truncated]"


def execute(model_path: Path, seed: int, variant: str) -> tuple[list[dict[str, Any]], str | None]:
    overall_deadline = time.monotonic() + OVERALL_TIMEOUT_SECONDS
    binary, binary_hash, build_seconds = build_binary(overall_deadline)
    trace(
        "binary_built",
        build_seconds=build_seconds,
        path=str(binary.relative_to(ROOT)),
        sha256=binary_hash,
    )
    port = free_loopback_port()
    run_dir = ROOT / ".qualification/serving" / f"{variant}-{os.getpid()}"
    adapter_dir = run_dir / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    environment = server_environment(variant, model_path, port, adapter_dir)
    policy_events_started = time.monotonic()
    process = subprocess.Popen(
        [str(binary), "--config", "/dev/null", "serve", "--served-model-id", MODEL_ID],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    server_log = ServerLog(process.stdout)
    server_log.start()
    sampler = MemorySampler(port)
    slow: SlowConsumer | None = None
    try:
        wait_ready(port, process, server_log, overall_deadline)
        health_before_warmup = read_stable_health(
            port, overall_deadline, "startup graph snapshot"
        )
        debug_start = json_request(port, "GET", "/v1/debug/model-state")
        failures = attest_runtime(variant, health_before_warmup, debug_start)
        if failures:
            raise QualificationError("startup runtime attestation failed: " + " | ".join(failures))
        trace("server_ready", port=port, variant=variant)

        warmup: StreamResult | None = None
        health_measurement_start: dict[str, Any] = {}
        expect_graphs = VARIANT_CONFIGS[variant]["runtime"]["rocm_graphs_enabled"]
        for attempt in range(MAX_WARMUP_REQUESTS):
            warmup = run_stream(
                port,
                name=f"warmup-{attempt + 1}",
                marker=f"QUAL-{variant}-{seed}-warmup-{attempt + 1}",
                prompt_words=16 + attempt * 8,
                max_tokens=WARMUP_MAX_TOKENS,
                seed=seed + attempt,
                absolute_deadline=overall_deadline,
            )
            if not warmup.success:
                raise QualificationError(
                    f"warmup request {attempt + 1} failed: "
                    f"{warmup.error or warmup.finish_reason}"
                )
            health_measurement_start = read_stable_health(
                port, overall_deadline, "post-warmup graph snapshot"
            )
            warmup_attestation = attest_runtime(
                variant, health_measurement_start, debug_start
            )
            if warmup_attestation:
                raise QualificationError(
                    "post-warmup runtime attestation failed: "
                    + " | ".join(warmup_attestation)
                )
            graph = graph_snapshot(health_measurement_start)
            if not expect_graphs or (
                graph["capture_successes"] >= 1
                and graph["replay_successes"] >= 1
                and graph["failures"] == 0
            ):
                break
        else:
            raise QualificationError(
                f"ROCm graph warmup did not capture and replay within {MAX_WARMUP_REQUESTS} requests"
            )
        assert warmup is not None
        measurement_started = time.monotonic()
        sampler.start()
        first_token = threading.Event()
        normal_word_counts = (16, 32, 64, 128, 256, 384, 512, 768)
        measured: list[StreamResult] = []
        cancellation_marker = f"QUAL-{variant}-{seed}-cancel"
        slow_marker = f"QUAL-{variant}-{seed}-slow"
        pressure_window: DeliveryPressureWindow | None = None
        delivery_pressure_observed = False
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=12)
        submitted: list[concurrent.futures.Future[StreamResult]] = []
        abort_workers = threading.Event()
        try:
            normal_futures = [
                pool.submit(
                    run_stream,
                    port,
                    name=f"normal-{index:02d}",
                    marker=f"QUAL-{variant}-{seed}-normal-{index:02d}",
                    prompt_words=normal_word_counts[index],
                    max_tokens=NORMAL_MAX_TOKENS,
                    seed=seed + 10 + index,
                    first_token_event=first_token,
                    absolute_deadline=overall_deadline,
                    abort_event=abort_workers,
                )
                for index in range(NORMAL_REQUESTS)
            ]
            submitted.extend(normal_futures)
            if not first_token.wait(
                timeout=remaining_until(
                    overall_deadline, "normal first token", REQUEST_TIMEOUT_SECONDS
                )
            ):
                raise QualificationError("normal decode did not produce a first token")
            long_future = pool.submit(
                run_stream,
                port,
                name="long-prefill",
                marker=f"QUAL-{variant}-{seed}-long",
                prompt_words=LONG_PREFILL_WORDS,
                max_tokens=LONG_PREFILL_MAX_TOKENS,
                seed=seed + 100,
                absolute_deadline=overall_deadline,
                abort_event=abort_workers,
            )
            cancel_future = pool.submit(
                run_stream,
                port,
                name="cancellation",
                marker=cancellation_marker,
                prompt_words=48,
                max_tokens=512,
                seed=seed + 101,
                cancel_after=CANCELLATION_AFTER_DELTAS,
                absolute_deadline=overall_deadline,
                abort_event=abort_workers,
            )
            submitted.extend((long_future, cancel_future))
            pressure_observed_since = time.monotonic()
            slow_pressure_baseline = batching_snapshot(
                json_request(port, "GET", "/health")
            )
            slow = SlowConsumer(port, slow_marker, seed + 102)
            slow.start()
            if not slow.header_received.wait(
                timeout=remaining_until(overall_deadline, "slow-consumer headers", 30.0)
            ):
                raise TimeoutError("slow consumer did not report response headers")
            if slow.error is not None:
                raise QualificationError(f"slow consumer failed: {slow.error}")
            pressure_peer_future = pool.submit(
                run_stream,
                port,
                name="pressure-peer",
                marker=f"QUAL-{variant}-{seed}-pressure-peer",
                prompt_words=PRESSURE_PEER_PROMPT_WORDS,
                max_tokens=PRESSURE_PEER_MAX_TOKENS,
                seed=seed + PRESSURE_PEER_SEED_OFFSET,
                absolute_deadline=overall_deadline,
                abort_event=abort_workers,
            )
            submitted.append(pressure_peer_future)
            pressure_window, delivery_pressure_observed, _ = wait_for_delivery_pressure(
                port,
                slow_pressure_baseline,
                server_log,
                f"qualification-{slow_marker}",
                pressure_observed_since,
                overall_deadline,
            )
            futures = [*normal_futures, long_future, pressure_peer_future]
            for future in futures:
                measured.append(
                    future.result(
                        timeout=remaining_until(overall_deadline, "mixed serving load")
                    )
                )
            cancellation = cancel_future.result(
                timeout=remaining_until(overall_deadline, "cancellation request")
            )
        finally:
            abort_workers.set()
            for future in submitted:
                future.cancel()
            _, unfinished = concurrent.futures.wait(
                submitted,
                timeout=max(0.0, min(10.0, overall_deadline - time.monotonic())),
            )
            pool.shutdown(wait=False, cancel_futures=True)
            if unfinished:
                raise QualificationError(
                    f"{len(unfinished)} request workers did not stop during cleanup"
                )
        if slow is None:
            raise QualificationError("slow consumer did not start")
        slow.close()
        sampler.close()
        if slow.error is not None:
            raise QualificationError(f"slow consumer failed: {slow.error}")
        if (
            not cancellation.cancelled
            or len(cancellation.semantic_times) < CANCELLATION_AFTER_DELTAS
        ):
            raise QualificationError(
                "cancellation client did not abort after "
                f"{CANCELLATION_AFTER_DELTAS} deltas: {cancellation}"
            )
        cancellation_confirmed, _ = wait_for_cancellation_and_drain(
            port, cancellation_marker, overall_deadline
        )
        health_end = read_stable_health(port, overall_deadline, "final graph snapshot")
        debug_end = json_request(port, "GET", "/v1/debug/model-state")
        final_attestation = attest_runtime(variant, health_end, debug_end)
        execution_attestation = attest_runtime_execution(
            variant, health_measurement_start, health_end
        )
        if process.poll() is not None:
            raise QualificationError(f"server exited during mixed load ({process.returncode})")
        long_prefill = next(result for result in measured if result.name == "long-prefill")
        pressure_peer = next(result for result in measured if result.name == "pressure-peer")
        slow_peer_success = int(
            healthy_peer_overlaps_pressure(pressure_peer, pressure_window)
        )
        measurement_events = server_log.events_since(measurement_started)
        policy_events = server_log.events_since(policy_events_started)
        values = metric_values(
            measured=measured,
            warmup=warmup,
            long_prefill=long_prefill,
            cancellation_confirmed=cancellation_confirmed,
            slow_peer_success=slow_peer_success,
            peak_memory=max(sampler.samples, default=0),
            health_after_warmup=health_measurement_start,
            health_measurement_start=health_measurement_start,
            health_end=health_end,
            events=measurement_events,
        )
        status_failures = [
            *final_attestation,
            *execution_attestation,
            *disabled_policy_attestation_failures(
                variant,
                policy_events,
                initial_blocks_total=batching_snapshot(health_before_warmup)[
                    "blocks_total"
                ],
                final_blocks_total=batching_snapshot(health_end)["blocks_total"],
            ),
        ]
        if values["request_failure_count"] != 0:
            status_failures.append(f"{values['request_failure_count']} measured requests failed")
        if values["zero_token_response_count"] != 0:
            status_failures.append(f"{values['zero_token_response_count']} responses had zero tokens")
        if values["batching_batched_decode_forward_count"] < 1:
            status_failures.append("measured load executed no batched decode forward")
        if values["batching_decode_row_count"] <= values["batching_decode_forward_count"]:
            status_failures.append("measured decode rows do not prove multi-row batching")
        if not cancellation_confirmed:
            status_failures.append("server did not confirm cancellation cleanup")
        if slow_peer_success < 1:
            status_failures.append(
                "the dedicated pressure peer did not emit actor-ready tokens before, "
                "inside, and after the attributed slow-consumer pressure window"
            )
        if not delivery_pressure_observed:
            status_failures.append(
                "slow consumer did not produce request-attributed backpressure and stall eviction"
            )
        if values["client_backpressure_event_count"] < 1:
            status_failures.append("no response-channel backpressure event was counted")
        if values["client_stall_eviction_count"] < 1:
            status_failures.append("no stalled response channel was evicted")
        if values["batching_total_errors"] != values["client_stall_eviction_count"]:
            status_failures.append(
                "batching errors were not exactly the intentional stalled-client evictions"
            )
        if values["graph_measured_capture_failure_count"] != 0:
            status_failures.append("ROCm graph capture failed during qualification")
        if values["graph_measured_replay_failure_count"] != 0:
            status_failures.append("ROCm graph replay failed during qualification")
        if values["unexplained_itl_outlier_count"] != 0:
            status_failures.append(
                f"{values['unexplained_itl_outlier_count']} ITL outliers were unexplained"
            )
        if values["attributed_itl_outlier_count"] != 0:
            status_failures.append(
                f"{values['attributed_itl_outlier_count']} healthy-request ITL outliers "
                "coincided with runtime events"
            )
        if values["response_queue_delay_ms_p999"] > OUTLIER_ABSOLUTE_MS:
            status_failures.append(
                "healthy response-channel queue delay exceeded the 250 ms stall threshold"
            )
        if not sampler.samples:
            status_failures.append("GPU memory sampler collected no values")
        if sampler.errors:
            status_failures.append("GPU memory sampler errors: " + ", ".join(sampler.errors))
        for result in [warmup, *measured, cancellation]:
            trace(
                "request_result",
                cancelled=result.cancelled,
                completion_tokens=result.completion_tokens,
                done=result.done,
                e2e_ms=result.e2e_ms,
                error=result.error,
                finish_reason=result.finish_reason,
                name=result.name,
                prompt_tokens=result.prompt_tokens,
                semantic_events=len(result.semantic_times),
                ttft_ms=result.ttft_ms,
            )
        details = " | ".join(status_failures) if status_failures else None
        return metrics_from_values(values), details
    finally:
        if slow is not None:
            slow.close()
        sampler.close()
        terminate_process(process)
        server_log.join()
        shutil.rmtree(run_dir, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    variant = os.environ.get(VARIANT_ENV, "")
    result_path_value = os.environ.get(RESULT_ENV)
    if not result_path_value:
        print(f"{RESULT_ENV} is required", file=sys.stderr)
        return 2
    result_path = Path(result_path_value)
    effective_config = VARIANT_CONFIGS.get(variant, {})
    status = "failed"
    details: str | None = None
    metrics = zero_metrics()
    try:
        if variant not in VARIANT_CONFIGS:
            raise QualificationError(
                f"{VARIANT_ENV} must name one of {sorted(VARIANT_CONFIGS)}, got {variant!r}"
            )
        model_path = args.model_path.resolve(strict=True)
        if not model_path.is_dir():
            raise QualificationError("--model-path must be a directory")
        metrics, details = execute(model_path, args.seed, variant)
        status = "passed" if details is None else "failed"
    except Exception as exc:
        details = f"{type(exc).__name__}: {exc}"
        trace("qualification_error", details=details)
    result = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "status": status,
        "duration_seconds": time.monotonic() - started,
        "effective_config": effective_config,
        "metrics": metrics,
        "tolerances": [],
        "details": bounded_details(details),
    }
    try:
        write_result(result_path, result)
    except Exception as exc:
        print(f"cannot write qualification result: {exc}", file=sys.stderr)
        return 2
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
