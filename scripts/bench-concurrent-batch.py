#!/usr/bin/env python3
"""Fail-closed OpenAI-compatible serving concurrency benchmark.

Measured requests use only ``POST /v1/chat/completions`` with streaming usage,
so the same driver and request bodies can be used for Kiln and vLLM.  Kiln's
``/health`` diagnostics are optional side evidence and never change the timed
request path.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "kiln.serving-benchmark.v1"
WORKLOAD_SCHEMA = "kiln.serving-benchmark-workload.v1"
DRIVER_VERSION = "2"
PROMPT_TEMPLATE_VERSION = "equal-token-multiset-v1"
ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_DIR = ROOT / "scripts" / "qualification"
if str(QUALIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(QUALIFICATION_DIR))

from strict_json import loads as strict_json_loads  # noqa: E402

PROMPT_MARKERS = (
    "amber",
    "birch",
    "cobalt",
    "delta",
    "ember",
    "frost",
    "granite",
    "harbor",
    "indigo",
    "juniper",
    "keystone",
    "linen",
    "maple",
    "nickel",
    "onyx",
    "prairie",
    "quartz",
    "raven",
    "silver",
    "timber",
    "umber",
    "violet",
    "willow",
    "zinc",
)

COUNTER_FIELDS = (
    "total_decode_forwards",
    "total_batched_decode_forwards",
    "total_decode_rows",
    "total_decode_tokens",
    "total_decode_forward_ms",
    "total_prefill_forwards",
    "total_prefill_tokens",
    "total_prefill_layers",
    "total_prefill_layer_yields",
    "total_prefill_forward_ms",
    "total_admission_calls",
    "total_admission_ms",
    "total_errors",
)


class BenchmarkError(RuntimeError):
    """A benchmark contract or preflight failure."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def text_sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def percentile_r7(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * probability
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def parse_sizes(raw: str) -> list[int]:
    try:
        sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise BenchmarkError(f"--sizes must contain decimal integers: {raw!r}") from exc
    if not sizes or any(size <= 0 or size > 4096 for size in sizes):
        raise BenchmarkError("--sizes must contain integers in 1..=4096")
    if sizes != sorted(set(sizes)):
        raise BenchmarkError("--sizes must be unique and strictly increasing")
    return sizes


def deterministic_prompt(run_id: str, phase: str, request_index: int) -> str:
    def marker_key(marker: str) -> bytes:
        material = f"{run_id}\0{phase}\0{request_index}\0{marker}".encode("utf-8")
        return hashlib.sha256(material).digest()

    markers = sorted(PROMPT_MARKERS, key=marker_key)
    marker_sequence = " | ".join(markers)
    return (
        "Write a detailed technical paragraph explaining why deterministic, "
        "reproducible performance measurements need controlled workloads, "
        "explicit error accounting, and tail-latency reporting. Continue until "
        "the response limit; do not mention these instructions.\n"
        f"Benchmark run: {run_id}; phase: {phase}.\n"
        f"Marker sequence: {marker_sequence}."
    )


def build_request_body(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "n": 1,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }


class SSEParser:
    def __init__(self) -> None:
        self._data_lines: list[str] = []

    def feed_line(self, line: str) -> list[str]:
        line = line.rstrip("\r\n")
        if not line:
            if not self._data_lines:
                return []
            data = "\n".join(self._data_lines)
            self._data_lines.clear()
            return [data]
        if line.startswith(":"):
            return []
        field, separator, value = line.partition(":")
        if field == "data":
            self._data_lines.append(value[1:] if separator and value.startswith(" ") else value)
        return []

    def finish(self) -> list[str]:
        if not self._data_lines:
            return []
        data = "\n".join(self._data_lines)
        self._data_lines.clear()
        return [data]


@dataclasses.dataclass
class RequestResult:
    index: int
    prompt_sha256: str
    started: float
    ended: float
    semantic_times: list[float]
    content: str
    reasoning_content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str | None
    done: bool
    error: str | None

    @property
    def ttft_ms(self) -> float | None:
        if not self.semantic_times:
            return None
        return (self.semantic_times[0] - self.started) * 1000.0

    @property
    def e2e_ms(self) -> float:
        return (self.ended - self.started) * 1000.0

    @property
    def itls_ms(self) -> list[float]:
        return [
            (current - previous) * 1000.0
            for previous, current in zip(self.semantic_times, self.semantic_times[1:])
        ]

    @property
    def output_sha256(self) -> str:
        return text_sha256(self.reasoning_content + "\x1e" + self.content)


def failed_result(index: int, prompt_sha256: str, started: float, exc: BaseException) -> RequestResult:
    return RequestResult(
        index=index,
        prompt_sha256=prompt_sha256,
        started=started,
        ended=time.perf_counter(),
        semantic_times=[],
        content="",
        reasoning_content="",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        finish_reason=None,
        done=False,
        error=f"{type(exc).__name__}: {exc}",
    )


def response_semantic_parts(value: dict[str, Any]) -> tuple[list[str], list[str]]:
    content: list[str] = []
    reasoning: list[str] = []
    choices = value.get("choices")
    if not isinstance(choices, list):
        return content, reasoning
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        part = delta.get("content")
        if isinstance(part, str) and part:
            content.append(part)
        for field in ("reasoning_content", "reasoning"):
            part = delta.get(field)
            if isinstance(part, str) and part:
                reasoning.append(part)
    return content, reasoning


def response_finish_reasons(value: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    choices = value.get("choices")
    if not isinstance(choices, list):
        return reasons
    for choice in choices:
        if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str):
            reasons.append(choice["finish_reason"])
    return reasons


def validate_usage(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, dict):
        raise BenchmarkError("usage must be an object")
    parsed: list[int] = []
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise BenchmarkError(f"usage.{field} must be a non-negative integer")
        parsed.append(item)
    if parsed[2] != parsed[0] + parsed[1]:
        raise BenchmarkError("usage.total_tokens does not equal prompt plus completion tokens")
    return parsed[0], parsed[1], parsed[2]


def stream_request(
    *,
    index: int,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout_secs: float,
    barrier: threading.Barrier,
) -> RequestResult:
    prompt = body["messages"][0]["content"]
    prompt_sha256 = text_sha256(prompt)
    started = time.perf_counter()
    try:
        barrier.wait(timeout=min(timeout_secs, 30.0))
        started = time.perf_counter()
        request = urllib.request.Request(
            url,
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        semantic_times: list[float] = []
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasons: list[str] = []
        usage_records = 0
        prompt_tokens = completion_tokens = total_tokens = 0
        done = False
        parser = SSEParser()
        with urllib.request.urlopen(request, timeout=timeout_secs) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type.lower():
                raise BenchmarkError(f"expected text/event-stream, got {content_type!r}")
            for raw_line in response:
                observed = time.perf_counter()
                for data in parser.feed_line(raw_line.decode("utf-8")):
                    if data == "[DONE]":
                        done = True
                        continue
                    value = strict_json_loads(data)
                    if not isinstance(value, dict):
                        raise BenchmarkError("SSE data payload must be an object")
                    content, reasoning = response_semantic_parts(value)
                    if content or reasoning:
                        semantic_times.append(observed)
                        content_parts.extend(content)
                        reasoning_parts.extend(reasoning)
                    reasons.extend(response_finish_reasons(value))
                    usage = value.get("usage")
                    if usage is not None:
                        usage_records += 1
                        if usage_records != 1:
                            raise BenchmarkError("stream emitted multiple usage records")
                        prompt_tokens, completion_tokens, total_tokens = validate_usage(usage)
            for data in parser.finish():
                if data == "[DONE]":
                    done = True
                else:
                    raise BenchmarkError("stream ended with an unterminated non-DONE SSE event")
        if not done:
            raise BenchmarkError("stream ended without [DONE]")
        if usage_records != 1:
            raise BenchmarkError("stream did not emit exactly one usage record")
        if prompt_tokens <= 0 or completion_tokens <= 0:
            raise BenchmarkError("stream reported zero prompt or completion tokens")
        if len(reasons) != 1:
            raise BenchmarkError(f"stream emitted {len(reasons)} finish reasons")
        if not semantic_times:
            raise BenchmarkError("stream emitted no client-visible semantic deltas")
        return RequestResult(
            index=index,
            prompt_sha256=prompt_sha256,
            started=started,
            ended=time.perf_counter(),
            semantic_times=semantic_times,
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=reasons[0],
            done=True,
            error=None,
        )
    except Exception as exc:
        if isinstance(exc, urllib.error.HTTPError):
            try:
                detail = exc.read(1024).decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            exc = BenchmarkError(f"HTTP {exc.code}: {detail}")
        return failed_result(index, prompt_sha256, started, exc)


def fetch_json(url: str, headers: dict[str, str], timeout_secs: float) -> dict[str, Any]:
    try:
        request = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=timeout_secs) as response:
            value = strict_json_loads(response.read())
    except Exception as exc:
        raise BenchmarkError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"{url} did not return a JSON object")
    return value


def batching_snapshot(health: dict[str, Any]) -> dict[str, Any]:
    runtime = health.get("decode_runtime")
    snapshot = runtime.get("batching_engine") if isinstance(runtime, dict) else None
    if not isinstance(snapshot, dict):
        raise BenchmarkError("diagnostics omit decode_runtime.batching_engine")
    for field in ("max_decode_batch", "max_observed_batch_size", *COUNTER_FIELDS):
        value = snapshot.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise BenchmarkError(f"invalid batching diagnostic {field}={value!r}")
    return snapshot


def batching_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "effective_max_decode_batch": int(after["max_decode_batch"]),
        "process_max_observed_batch": int(after["max_observed_batch_size"]),
    }
    for field in COUNTER_FIELDS:
        if after[field] < before[field]:
            raise BenchmarkError(
                f"batching counter {field} regressed from {before[field]} to {after[field]}"
            )
        result[field] = after[field] - before[field]
    forwards = result["total_decode_forwards"]
    result["mean_decode_rows_per_forward"] = (
        result["total_decode_rows"] / forwards if forwards else 0.0
    )
    result["batched_decode_forward_fraction"] = (
        result["total_batched_decode_forwards"] / forwards if forwards else 0.0
    )
    return result


class MemorySampler:
    def __init__(self, path: Path | None, interval_ms: int) -> None:
        self.path = path
        self.interval_secs = interval_ms / 1000.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._baseline: int | None = None
        self._peak: int | None = None
        self._samples = 0

    def _read(self) -> int:
        if self.path is None:
            raise BenchmarkError("memory sampler is disabled")
        try:
            raw = self.path.read_text(encoding="utf-8").strip()
            value = int(raw)
        except (OSError, ValueError) as exc:
            raise BenchmarkError(f"cannot read memory counter {self.path}: {exc}") from exc
        if value < 0:
            raise BenchmarkError(f"memory counter at {self.path} is negative")
        return value

    def start(self) -> None:
        if self.path is None:
            return
        self.reset()
        self._thread = threading.Thread(target=self._run, name="benchmark-memory", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_secs):
            try:
                value = self._read()
            except Exception:
                continue
            with self._lock:
                self._peak = value if self._peak is None else max(self._peak, value)
                self._samples += 1

    def reset(self) -> None:
        if self.path is None:
            return
        value = self._read()
        with self._lock:
            self._baseline = value
            self._peak = value
            self._samples = 1

    def snapshot(self) -> dict[str, int] | None:
        if self.path is None:
            return None
        value = self._read()
        with self._lock:
            peak = max(self._peak or value, value)
            baseline = self._baseline or value
            samples = self._samples + 1
        return {
            "baseline_bytes": baseline,
            "peak_bytes": peak,
            "peak_delta_bytes": max(0, peak - baseline),
            "samples": samples,
        }

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def resolve_memory_path(raw: str) -> Path | None:
    if raw == "none":
        return None
    if raw != "auto":
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise BenchmarkError(f"memory counter does not exist: {path}")
        return path
    candidates = sorted(
        Path(path).resolve()
        for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")
        if Path(path).is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    raise BenchmarkError(
        "multiple DRM memory counters found; select one with --memory-path: "
        + ", ".join(str(path) for path in candidates)
    )


def gate(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def summarize_run(
    *,
    concurrency: int,
    repeat: int,
    elapsed_s: float,
    results: list[RequestResult],
    max_tokens: int,
    require_max_tokens: bool,
    require_uniform_prompt_tokens: bool,
    max_dispatch_spread_ms: float,
    slo_ttft_ms: float,
    slo_itl_ms: float,
    slo_e2e_ms: float,
    memory: dict[str, int] | None,
    require_memory: bool,
    server: dict[str, Any] | None,
    diagnostics_error: str | None,
) -> dict[str, Any]:
    successes = [result for result in results if result.error is None]
    errors = [
        {"index": result.index, "error": result.error}
        for result in results
        if result.error is not None
    ]
    ttfts = [result.ttft_ms for result in successes if result.ttft_ms is not None]
    e2es = [result.e2e_ms for result in successes]
    itls = [itl for result in successes for itl in result.itls_ms]
    completion_tokens = sum(result.completion_tokens for result in successes)
    prompt_token_values = [result.prompt_tokens for result in successes]
    dispatch_times = [result.started for result in results]
    dispatch_spread_ms = (
        (max(dispatch_times) - min(dispatch_times)) * 1000.0 if dispatch_times else 0.0
    )
    good_results = [
        result
        for result in successes
        if result.ttft_ms is not None
        and result.ttft_ms <= slo_ttft_ms
        and result.e2e_ms <= slo_e2e_ms
        and (not result.itls_ms or max(result.itls_ms) <= slo_itl_ms)
    ]
    gates = [
        gate(
            "all_requests_succeeded",
            len(successes) == concurrency,
            f"{len(successes)}/{concurrency} requests succeeded",
        ),
        gate(
            "positive_completion_usage",
            len(successes) == concurrency
            and all(result.completion_tokens > 0 for result in successes),
            "every measured request must contribute positive completion usage",
        ),
        gate(
            "dispatch_spread",
            dispatch_spread_ms <= max_dispatch_spread_ms,
            f"{dispatch_spread_ms:.3f} ms <= {max_dispatch_spread_ms:.3f} ms",
        ),
        gate(
            "diagnostics_readable",
            diagnostics_error is None,
            diagnostics_error or "not requested or parsed successfully",
        ),
    ]
    if require_max_tokens:
        exact = len(successes) == concurrency and all(
            result.completion_tokens == max_tokens and result.finish_reason == "length"
            for result in successes
        )
        gates.append(
            gate(
                "fixed_output_length",
                exact,
                f"every request must finish by length with exactly {max_tokens} tokens",
            )
        )
    if require_uniform_prompt_tokens:
        gates.append(
            gate(
                "uniform_prompt_tokens",
                len(prompt_token_values) == concurrency
                and len(set(prompt_token_values)) == 1,
                f"observed prompt-token counts: {sorted(set(prompt_token_values))}",
            )
        )
    if require_memory:
        gates.append(
            gate(
                "memory_measured",
                memory is not None and memory["samples"] >= 2,
                "a local device-memory counter must be sampled during the run",
            )
        )
    if server is not None:
        gates.append(
            gate(
                "server_reported_no_errors",
                server["total_errors"] == 0,
                f"batching-engine error delta: {server['total_errors']}",
            )
        )
    output_rows = [
        {
            "index": result.index,
            "output_sha256": result.output_sha256,
            "completion_tokens": result.completion_tokens,
            "finish_reason": result.finish_reason,
        }
        for result in sorted(successes, key=lambda result: result.index)
    ]
    prompt_rows = [
        {"index": result.index, "prompt_sha256": result.prompt_sha256}
        for result in sorted(results, key=lambda result: result.index)
    ]
    passed = all(item["passed"] for item in gates)
    return {
        "concurrency": concurrency,
        "repeat": repeat,
        "verdict": "passed" if passed else "failed",
        "elapsed_s": elapsed_s,
        "request_count": concurrency,
        "success_count": len(successes),
        "error_count": len(errors),
        "errors": errors,
        "prompt_tokens_min": min(prompt_token_values) if prompt_token_values else 0,
        "prompt_tokens_max": max(prompt_token_values) if prompt_token_values else 0,
        "completion_tokens": completion_tokens,
        "request_throughput_per_s": len(successes) / elapsed_s if elapsed_s else 0.0,
        "output_token_throughput_per_s": completion_tokens / elapsed_s if elapsed_s else 0.0,
        "slo_good_request_count": len(good_results),
        "slo_goodput_requests_per_s": len(good_results) / elapsed_s if elapsed_s else 0.0,
        "slo_goodput_tokens_per_s": (
            sum(result.completion_tokens for result in good_results) / elapsed_s
            if elapsed_s
            else 0.0
        ),
        "dispatch_spread_ms": dispatch_spread_ms,
        "ttft_ms_p50": percentile_r7(ttfts, 0.50),
        "ttft_ms_p99": percentile_r7(ttfts, 0.99),
        "ttft_ms_p999": percentile_r7(ttfts, 0.999),
        "e2e_ms_p50": percentile_r7(e2es, 0.50),
        "e2e_ms_p99": percentile_r7(e2es, 0.99),
        "e2e_ms_p999": percentile_r7(e2es, 0.999),
        "client_visible_itl_ms_p50": percentile_r7(itls, 0.50),
        "client_visible_itl_ms_p99": percentile_r7(itls, 0.99),
        "client_visible_itl_ms_p999": percentile_r7(itls, 0.999),
        "client_visible_stream_event_count": sum(
            len(result.semantic_times) for result in successes
        ),
        "prompt_set_sha256": canonical_sha256(prompt_rows),
        "output_set_sha256": canonical_sha256(output_rows),
        "memory": memory,
        "server": server,
        "gates": gates,
    }


def run_once(
    *,
    args: argparse.Namespace,
    concurrency: int,
    repeat: int,
    max_tokens: int,
    phase: str,
    headers: dict[str, str],
    sampler: MemorySampler,
    diagnostics_url: str | None,
) -> dict[str, Any]:
    bodies: list[dict[str, Any]] = []
    prompts: set[str] = set()
    for index in range(concurrency):
        prompt = deterministic_prompt(args.run_id, phase, index)
        if prompt in prompts:
            raise BenchmarkError("deterministic prompt construction produced a duplicate")
        prompts.add(prompt)
        bodies.append(
            build_request_body(
                model=args.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + index,
                enable_thinking=args.enable_thinking,
            )
        )

    diagnostics_before: dict[str, Any] | None = None
    diagnostics_error: str | None = None
    if diagnostics_url is not None:
        try:
            diagnostics_before = batching_snapshot(
                fetch_json(diagnostics_url, headers, args.timeout_secs)
            )
        except Exception as exc:
            diagnostics_error = f"before run: {type(exc).__name__}: {exc}"

    sampler.reset()
    barrier = threading.Barrier(concurrency + 1)
    results: list[RequestResult | None] = [None] * concurrency

    def worker(index: int) -> None:
        results[index] = stream_request(
            index=index,
            url=f"{args.base_url}/v1/chat/completions",
            body=bodies[index],
            headers=headers,
            timeout_secs=args.timeout_secs,
            barrier=barrier,
        )

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(concurrency)
    ]
    for thread in threads:
        thread.start()
    wall_started = time.perf_counter()
    try:
        barrier.wait(timeout=min(args.timeout_secs, 30.0))
    except threading.BrokenBarrierError as exc:
        raise BenchmarkError("request launch barrier broke before dispatch") from exc
    join_deadline = time.monotonic() + args.timeout_secs + 5.0
    for thread in threads:
        thread.join(timeout=max(0.0, join_deadline - time.monotonic()))
    wall_ended = time.perf_counter()
    if any(thread.is_alive() for thread in threads):
        raise BenchmarkError("one or more request workers exceeded the join deadline")

    typed_results = [result for result in results if result is not None]
    if len(typed_results) != concurrency:
        raise BenchmarkError("one or more request workers exited without publishing a result")

    server_delta: dict[str, Any] | None = None
    if diagnostics_url is not None and diagnostics_error is None:
        try:
            diagnostics_after = batching_snapshot(
                fetch_json(diagnostics_url, headers, args.timeout_secs)
            )
            assert diagnostics_before is not None
            server_delta = batching_delta(diagnostics_before, diagnostics_after)
        except Exception as exc:
            diagnostics_error = f"after run: {type(exc).__name__}: {exc}"

    return summarize_run(
        concurrency=concurrency,
        repeat=repeat,
        elapsed_s=wall_ended - wall_started,
        results=typed_results,
        max_tokens=max_tokens,
        require_max_tokens=args.require_max_tokens,
        require_uniform_prompt_tokens=args.require_uniform_prompt_tokens,
        max_dispatch_spread_ms=args.max_dispatch_spread_ms,
        slo_ttft_ms=args.slo_ttft_ms,
        slo_itl_ms=args.slo_itl_ms,
        slo_e2e_ms=args.slo_e2e_ms,
        memory=sampler.snapshot(),
        require_memory=args.require_memory,
        server=server_delta,
        diagnostics_error=diagnostics_error,
    )


def compare_reference(receipt: dict[str, Any], reference_path: Path) -> dict[str, Any]:
    try:
        reference_bytes = reference_path.read_bytes()
        reference = strict_json_loads(reference_bytes)
    except Exception as exc:
        raise BenchmarkError(
            f"cannot load reference receipt {reference_path}: {type(exc).__name__}: {exc}"
        ) from exc
    if reference.get("schema") != SCHEMA:
        raise BenchmarkError(f"reference receipt does not use {SCHEMA}")
    if reference.get("workload_fingerprint") != receipt.get("workload_fingerprint"):
        raise BenchmarkError("reference receipt has a different workload fingerprint")
    current_rows = {
        (row["concurrency"], row["repeat"]): row for row in receipt.get("runs", [])
    }
    reference_rows = {
        (row["concurrency"], row["repeat"]): row for row in reference.get("runs", [])
    }
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(current_rows) | set(reference_rows)):
        current = current_rows.get(key)
        expected = reference_rows.get(key)
        if current is None or expected is None:
            mismatches.append({"concurrency": key[0], "repeat": key[1], "reason": "missing_run"})
        elif current["prompt_set_sha256"] != expected["prompt_set_sha256"]:
            mismatches.append(
                {"concurrency": key[0], "repeat": key[1], "reason": "prompt_mismatch"}
            )
        elif current["output_set_sha256"] != expected["output_set_sha256"]:
            mismatches.append(
                {"concurrency": key[0], "repeat": key[1], "reason": "output_mismatch"}
            )
    return {
        "reference_receipt_sha256": "sha256:" + hashlib.sha256(reference_bytes).hexdigest(),
        "reference_engine": reference.get("engine"),
        "matched": not mismatches,
        "mismatches": mismatches,
    }


def command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def repository_identity() -> dict[str, Any]:
    commit = command_output(["git", "rev-parse", "HEAD"])
    status = command_output(["git", "status", "--porcelain"])
    source_hash = command_output(
        [sys.executable, str(ROOT / "scripts" / "qualification" / "source_tree_hash.py")]
    )
    return {
        "commit": commit,
        "dirty": bool(status),
        "source_tree_sha256": source_hash,
    }


def probe_models(base_url: str, headers: dict[str, str], timeout_secs: float) -> list[str]:
    value = fetch_json(f"{base_url}/v1/models", headers, timeout_secs)
    data = value.get("data")
    if not isinstance(data, list):
        raise BenchmarkError("/v1/models response omits data array")
    models = [row.get("id") for row in data if isinstance(row, dict) and isinstance(row.get("id"), str)]
    if not models:
        raise BenchmarkError("/v1/models returned no model identifiers")
    return models


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise BenchmarkError(f"refusing to overwrite existing receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def workload_contract(args: argparse.Namespace, sizes: list[int]) -> dict[str, Any]:
    return {
        "schema": WORKLOAD_SCHEMA,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "run_id": args.run_id,
        "model": args.model,
        "endpoint": "/v1/chat/completions",
        "stream": True,
        "stream_include_usage": True,
        "concurrency": sizes,
        "repeats": args.repeats,
        "warmup_requests": args.warmup_requests,
        "max_tokens": args.max_tokens,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "seed": args.seed,
        },
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        "arrival_pattern": "thread_barrier_all_at_once",
        "require_max_tokens": args.require_max_tokens,
        "require_uniform_prompt_tokens": args.require_uniform_prompt_tokens,
        "max_dispatch_spread_ms": args.max_dispatch_spread_ms,
        "slo": {
            "ttft_ms": args.slo_ttft_ms,
            "client_visible_itl_ms": args.slo_itl_ms,
            "e2e_ms": args.slo_e2e_ms,
        },
    }


def print_run(row: dict[str, Any]) -> None:
    server = row.get("server") or {}
    width = server.get("process_max_observed_batch")
    mean = server.get("mean_decode_rows_per_forward")
    width_text = "n/a" if width is None else str(width)
    mean_text = "n/a" if mean is None else f"{mean:.2f}"
    print(
        f"[c={row['concurrency']:>3} r={row['repeat']}] {row['verdict']:<6} "
        f"tok/s={row['output_token_throughput_per_s']:.2f} "
        f"good_tok/s={row['slo_goodput_tokens_per_s']:.2f} "
        f"ttft_p99={row['ttft_ms_p99'] or 0.0:.1f}ms "
        f"itl_p99={row['client_visible_itl_ms_p99'] or 0.0:.1f}ms "
        f"batch_max={width_text} batch_mean={mean_text} "
        f"ok={row['success_count']}/{row['request_count']}",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("kiln", "vllm"), default="kiln")
    parser.add_argument("--base-url", "--host", dest="base_url", default="http://127.0.0.1:8420")
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument("--runtime-identity")
    parser.add_argument(
        "--run-id",
        default="manual-v1",
        help="Shared deterministic ID for both engine runs",
    )
    parser.add_argument("--sizes", default="1,8,16,32,64")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--warmup", action="store_true", help="Compatibility alias; warmup is already on")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--require-max-tokens", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-uniform-prompt-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-dispatch-spread-ms", type=float, default=250.0)
    parser.add_argument("--slo-ttft-ms", type=float, default=5_000.0)
    parser.add_argument("--slo-itl-ms", type=float, default=250.0)
    parser.add_argument("--slo-e2e-ms", type=float, default=60_000.0)
    parser.add_argument("--timeout-secs", type=float, default=600.0)
    parser.add_argument("--diagnostics-url", default="auto")
    parser.add_argument("--memory-path", default="auto")
    parser.add_argument("--memory-sample-ms", type=int, default=50)
    parser.add_argument("--require-memory", action="store_true")
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument(
        "--api-key",
        help="Explicit bearer token (prefer --api-key-env to keep it out of process listings)",
    )
    authentication.add_argument(
        "--api-key-env",
        help="Name of the environment variable containing the bearer token",
    )
    parser.add_argument("--reference-receipt", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--mode",
        choices=("concurrent",),
        default="concurrent",
        help="Compatibility flag; only the engine-neutral concurrent path is supported",
    )
    args = parser.parse_args(argv)
    args.base_url = args.base_url.rstrip("/")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", args.run_id):
        parser.error("run-id must be 3..128 portable identifier characters")
    if not 0 < args.repeats <= 1000 or not 0 < args.max_tokens <= 2**31:
        parser.error("repeats must be in 1..=1000 and max-tokens in 1..=2^31")
    if not 0 <= args.warmup_requests <= 4096:
        parser.error("warmup-requests must be in 0..=4096")
    if not math.isfinite(args.temperature) or not math.isfinite(args.top_p):
        parser.error("temperature and top-p must be finite")
    if not 0.0 <= args.temperature or not 0.0 < args.top_p <= 1.0:
        parser.error("temperature must be non-negative and top-p must be in (0, 1]")
    finite_positive = (
        args.timeout_secs,
        args.max_dispatch_spread_ms,
        args.slo_ttft_ms,
        args.slo_itl_ms,
        args.slo_e2e_ms,
    )
    if any(not math.isfinite(value) or value <= 0 for value in finite_positive):
        parser.error("timeouts, dispatch limit, and SLO thresholds must be finite and positive")
    if args.memory_sample_ms <= 0:
        parser.error("memory sampling cadence must be positive")
    if not 0 <= args.seed <= 2**64 - 1:
        parser.error("seed must fit an unsigned 64-bit integer")
    args.api_key_source = "none"
    if args.api_key_env is not None:
        args.api_key = os.environ.get(args.api_key_env)
        if not args.api_key:
            parser.error(f"api-key environment variable {args.api_key_env!r} is unset or empty")
        args.api_key_source = "environment"
    elif args.api_key is not None:
        if not args.api_key:
            parser.error("api-key cannot be empty")
        args.api_key_source = "argument"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.out is not None and args.out.exists():
            raise BenchmarkError(f"refusing to overwrite existing receipt: {args.out}")
        sizes = parse_sizes(args.sizes)
        largest_seed_offset = max([*sizes, args.warmup_requests]) - 1
        if args.seed + largest_seed_offset > 2**64 - 1:
            raise BenchmarkError("seed plus the largest request index exceeds u64")
        repo = repository_identity()
        if repo["dirty"] and not args.allow_dirty:
            raise BenchmarkError("repository is dirty; commit first or use --allow-dirty for a diagnostic")
        if args.runtime_identity is None:
            if args.engine == "kiln" and repo["commit"] and not repo["dirty"]:
                args.runtime_identity = f"kiln-git:{repo['commit']}"
            else:
                raise BenchmarkError("--runtime-identity is required for this engine/source state")

        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": f"kiln-serving-benchmark/{DRIVER_VERSION}",
        }
        if args.api_key:
            headers["Authorization"] = f"Bearer {args.api_key}"
        models = probe_models(args.base_url, headers, args.timeout_secs)
        if args.model not in models:
            raise BenchmarkError(
                f"requested model {args.model!r} is absent from /v1/models: {models}"
            )

        diagnostics_url: str | None
        if args.diagnostics_url == "none":
            diagnostics_url = None
        elif args.diagnostics_url == "auto":
            diagnostics_url = f"{args.base_url}/health" if args.engine == "kiln" else None
        else:
            diagnostics_url = args.diagnostics_url

        memory_path = resolve_memory_path(args.memory_path)
        sampler = MemorySampler(memory_path, args.memory_sample_ms)
        sampler.start()
        try:
            warmup: dict[str, Any] | None = None
            if args.warmup_requests:
                warmup = run_once(
                    args=args,
                    concurrency=args.warmup_requests,
                    repeat=-1,
                    max_tokens=min(16, args.max_tokens),
                    phase=f"warmup-c{args.warmup_requests:03d}",
                    headers=headers,
                    sampler=sampler,
                    diagnostics_url=diagnostics_url,
                )
                print(f"[warmup] {warmup['verdict']} ok={warmup['success_count']}/{warmup['request_count']}")

            runs: list[dict[str, Any]] = []
            if warmup is None or warmup["verdict"] == "passed":
                for concurrency in sizes:
                    for repeat in range(args.repeats):
                        row = run_once(
                            args=args,
                            concurrency=concurrency,
                            repeat=repeat,
                            max_tokens=args.max_tokens,
                            phase=f"measure-c{concurrency:03d}-r{repeat:03d}",
                            headers=headers,
                            sampler=sampler,
                            diagnostics_url=diagnostics_url,
                        )
                        runs.append(row)
                        print_run(row)
        finally:
            sampler.stop()

        workload = workload_contract(args, sizes)
        health_version = None
        if diagnostics_url is not None:
            try:
                health_version = fetch_json(diagnostics_url, headers, args.timeout_secs).get("version")
            except Exception:
                pass
        receipt: dict[str, Any] = {
            "schema": SCHEMA,
            "driver_version": DRIVER_VERSION,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "engine": {
                "name": args.engine,
                "runtime_identity": args.runtime_identity,
                "reported_version": health_version,
                "base_url": args.base_url,
                "model": args.model,
                "available_models": models,
                "authentication_configured": bool(args.api_key),
                "authentication_source": args.api_key_source,
            },
            "driver_environment": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "repository": repo,
            },
            "workload": workload,
            "workload_fingerprint": canonical_sha256(workload),
            "memory_sampler": {
                "source": "drm_vram_used" if memory_path is not None else "unavailable",
                "path": str(memory_path) if memory_path is not None else None,
                "interval_ms": args.memory_sample_ms if memory_path is not None else None,
            },
            "diagnostics": {
                "url": diagnostics_url,
                "timed_request_path_affected": False,
            },
            "warmup": warmup,
            "runs": runs,
        }
        if args.reference_receipt is not None:
            receipt["comparison"] = compare_reference(receipt, args.reference_receipt)
        passed = (
            (warmup is None or warmup["verdict"] == "passed")
            and len(runs) == len(sizes) * args.repeats
            and all(row["verdict"] == "passed" for row in runs)
            and receipt.get("comparison", {}).get("matched", True)
        )
        receipt["verdict"] = "passed" if passed else "failed"
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        if args.out is not None:
            atomic_write_json(args.out, receipt)
            print(f"wrote {args.out}")
        else:
            print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if passed else 2
    except BenchmarkError as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
