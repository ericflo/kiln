#!/usr/bin/env python3
"""Qualify full-model ROCm eager output against warmed HIP-graph replay."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import http.client
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import serve_mixed_load as mixed


ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "rocm-eager-warmed-graph-correctness"
VARIANT_ID = "eager-vs-warmed-graph"
RESULT_ENV = mixed.RESULT_ENV
VARIANT_ENV = mixed.VARIANT_ENV
GRAPH_CACHE_MAX = 12
REQUEST_TIMEOUT_SECONDS = 240.0
OVERALL_TIMEOUT_SECONDS = 1200.0
MAX_TOKENS = 48
SCENARIOS = (
    ("short", 16),
    ("medium", 96),
    ("long", 384),
    ("bucket-crossing", 768),
)


class CorrectnessError(RuntimeError):
    pass


def mode_config(graphs: bool) -> dict[str, Any]:
    value = mixed._variant_config(
        serving_profile="experimental",
        kv_autoscale_requested=False,
        kv_autoscale_enabled=False,
        memory_reclaim_requested_mode="off",
        memory_reclaim_mode="off",
        rocm_graphs_requested=graphs,
        rocm_graphs_enabled=graphs,
    )
    value["server"]["deterministic"] = True
    value["server"]["max_decode_batch"] = 1
    value["server"]["max_active_requests"] = 1
    value["server"]["max_prefill_staging_slots"] = 0
    value["server"]["max_prefill_staging_priority_burst"] = 0
    value["workload"] = {
        "comparison": "exact_action_tokens_and_selected_logprobs",
        "graph_cache_max": GRAPH_CACHE_MAX,
        "max_tokens": MAX_TOKENS,
        "request_timeout_seconds": int(REQUEST_TIMEOUT_SECONDS),
        "scenarios": {name: words for name, words in SCENARIOS},
        "warm_then_measure": True,
    }
    return value


MODE_CONFIGS = {
    "eager": mode_config(False),
    "graph": mode_config(True),
}
mixed.VARIANT_CONFIGS.update(MODE_CONFIGS)
EFFECTIVE_CONFIG = {
    "build": MODE_CONFIGS["graph"]["build"],
    "comparison": MODE_CONFIGS["graph"]["workload"],
    "eager": {
        "runtime": MODE_CONFIGS["eager"]["runtime"],
        "server": MODE_CONFIGS["eager"]["server"],
    },
    "graph": {
        "runtime": MODE_CONFIGS["graph"]["runtime"],
        "server": MODE_CONFIGS["graph"]["server"],
    },
}


METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    "action_token_count": ("tokens", "sum", False),
    "behavior_logprob_mismatch_count": ("count", "sum", True),
    "eager_graph_activity_count": ("count", "sum", True),
    "eager_repeat_mismatch_count": ("count", "sum", True),
    "eager_request_count": ("count", "sum", False),
    "graph_capture_failure_count": ("count", "sum", True),
    "graph_capture_warm_count": ("count", "sum", False),
    "graph_fallback_count": ("count", "sum", True),
    "graph_measure_capture_count": ("count", "sum", True),
    "graph_measure_replay_count": ("count", "sum", False),
    "graph_replay_failure_count": ("count", "sum", True),
    "graph_repeat_mismatch_count": ("count", "sum", True),
    "graph_request_count": ("count", "sum", False),
    "graph_retained_count_end": ("graphs", "exact", False),
    "graph_slot_active_count_end": ("slots", "exact", True),
    "graph_slot_count_end": ("slots", "exact", False),
    "graph_slot_idle_count_end": ("slots", "exact", False),
    "graph_slot_reuse_count": ("count", "sum", False),
    "non_finite_logprob_count": ("count", "sum", True),
    "output_mismatch_count": ("count", "sum", True),
    "prefix_cache_hit_count": ("count", "sum", False),
    "request_failure_count": ("count", "sum", True),
    "scenario_count": ("count", "exact", False),
    "shutdown_forced_count": ("count", "sum", True),
    "shutdown_nonzero_count": ("count", "sum", True),
    "snapshot_residue_count": ("count", "sum", True),
    "token_id_mismatch_count": ("count", "sum", True),
}


@dataclasses.dataclass(frozen=True)
class CompletionRecord:
    scenario: str
    semantic: dict[str, Any]
    action_tokens: tuple[tuple[int, int, str, float | None], ...]
    sampled_logprobs: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class ModeRun:
    mode: str
    warm: tuple[CompletionRecord, ...]
    measured: tuple[CompletionRecord, ...]
    graph_before_measure: dict[str, int]
    graph_after_measure: dict[str, int]
    prefix_hits: int
    shutdown: mixed.ShutdownOutcome
    snapshot_residue: tuple[str, ...]


def require_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CorrectnessError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise CorrectnessError(f"{label} is not a canonical SHA-256 identity")
    return value


def parse_completion(value: Any, scenario: str, seed: int) -> CompletionRecord:
    if not isinstance(value, dict):
        raise CorrectnessError(f"{scenario} response is not an object")
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise CorrectnessError(f"{scenario} response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise CorrectnessError(f"{scenario} response has no canonical message content")
    finish_reason = choice.get("finish_reason")
    if finish_reason not in {"length", "stop"}:
        raise CorrectnessError(f"{scenario} finish_reason={finish_reason!r}")
    usage = value.get("usage")
    if not isinstance(usage, dict):
        raise CorrectnessError(f"{scenario} response usage is missing")
    prompt_tokens = require_int(usage.get("prompt_tokens"), f"{scenario} prompt_tokens", 1)
    completion_tokens = require_int(
        usage.get("completion_tokens"), f"{scenario} completion_tokens", 1
    )
    total_tokens = require_int(usage.get("total_tokens"), f"{scenario} total_tokens", 2)
    if total_tokens != prompt_tokens + completion_tokens:
        raise CorrectnessError(f"{scenario} usage token totals are inconsistent")

    provenance = choice.get("rollout_provenance")
    if not isinstance(provenance, dict):
        raise CorrectnessError(f"{scenario} rollout provenance is missing")
    if provenance.get("schema") != "kiln.rollout-provenance.v1":
        raise CorrectnessError(f"{scenario} rollout provenance schema is unsupported")
    if require_int(provenance.get("seed"), f"{scenario} provenance seed") != seed:
        raise CorrectnessError(f"{scenario} rollout seed does not match the request")
    if provenance.get("generation_backend") != "rocm":
        raise CorrectnessError(f"{scenario} generation backend is not ROCm")
    if (
        require_int(
            provenance.get("prompt_token_count"),
            f"{scenario} provenance prompt_token_count",
            1,
        )
        != prompt_tokens
    ):
        raise CorrectnessError(f"{scenario} provenance prompt count disagrees with usage")
    require_sha256(provenance.get("prompt_messages_sha256"), "prompt_messages_sha256")
    require_sha256(provenance.get("scored_payload_sha256"), "scored_payload_sha256")
    input_token_ids = provenance.get("input_token_ids")
    if not isinstance(input_token_ids, list) or not input_token_ids:
        raise CorrectnessError(f"{scenario} provenance input_token_ids are missing")
    normalized_input_ids = [
        require_int(token_id, f"{scenario} input token", 0) for token_id in input_token_ids
    ]

    raw_actions = provenance.get("action_tokens")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise CorrectnessError(f"{scenario} provenance action_tokens are missing")
    actions: list[tuple[int, int, str, float | None]] = []
    sampled_logprobs: list[float] = []
    for index, raw in enumerate(raw_actions):
        if not isinstance(raw, dict):
            raise CorrectnessError(f"{scenario} action {index} is not an object")
        sequence_index = require_int(
            raw.get("sequence_index"), f"{scenario} action sequence_index"
        )
        token_id = require_int(raw.get("token_id"), f"{scenario} action token_id")
        source = raw.get("source")
        if source not in {"sampled", "forced"}:
            raise CorrectnessError(f"{scenario} action source={source!r}")
        logprob = raw.get("behavior_logprob")
        if source == "sampled":
            if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
                raise CorrectnessError(f"{scenario} sampled action has no log-probability")
            normalized_logprob = float(logprob)
            if not math.isfinite(normalized_logprob) or normalized_logprob > 0:
                raise CorrectnessError(
                    f"{scenario} sampled action log-probability is invalid: {logprob!r}"
                )
            sampled_logprobs.append(normalized_logprob)
            logprob = normalized_logprob
        elif logprob is not None:
            raise CorrectnessError(f"{scenario} forced action carries a log-probability")
        if sequence_index >= len(normalized_input_ids) or normalized_input_ids[sequence_index] != token_id:
            raise CorrectnessError(f"{scenario} action {index} does not match input_token_ids")
        actions.append((sequence_index, token_id, source, logprob))
    if not sampled_logprobs:
        raise CorrectnessError(f"{scenario} provenance contains no sampled action")

    semantic = {
        "content": message["content"],
        "reasoning_content": message.get("reasoning_content"),
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "input_token_ids": normalized_input_ids,
        "prompt_token_count": prompt_tokens,
        "prompt_messages_sha256": provenance["prompt_messages_sha256"],
        "scored_payload_sha256": provenance["scored_payload_sha256"],
        "action_tokens": [list(action) for action in actions],
        "behavior_policy": provenance.get("behavior_policy"),
        "tokenizer": provenance.get("tokenizer"),
        "template_invocation": provenance.get("template_invocation", {}),
        "sampling": provenance.get("sampling"),
        "seed": seed,
        "generation_backend": "rocm",
    }
    return CompletionRecord(
        scenario=scenario,
        semantic=semantic,
        action_tokens=tuple(actions),
        sampled_logprobs=tuple(sampled_logprobs),
    )


def completion_request(port: int, scenario: str, prompt: str, seed: int) -> CompletionRecord:
    body = {
        "model": mixed.MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "adapter": None,
        "max_tokens": MAX_TOKENS,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 16,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": seed,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "rollout_provenance": True,
    }
    connection = http.client.HTTPConnection(
        "127.0.0.1", port, timeout=REQUEST_TIMEOUT_SECONDS
    )
    try:
        payload = json.dumps(body, separators=(",", ":"))
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "kiln-rocm-graph-correctness/1",
            },
        )
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise CorrectnessError(
                f"{scenario} returned HTTP {response.status}: {raw[:1024]!r}"
            )
        return parse_completion(json.loads(raw), scenario, seed)
    finally:
        connection.close()


def prefix_snapshot(health: dict[str, Any]) -> dict[str, int]:
    raw = health.get("prefix_cache")
    if not isinstance(raw, dict):
        raise CorrectnessError("health.prefix_cache is missing")
    return {
        field: require_int(raw.get(field), f"prefix_cache.{field}")
        for field in (
            "lookup_hits",
            "cached_blocks",
            "active_leases",
            "pending_release_entries",
        )
    }


def wait_drained(port: int, deadline: float, label: str) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        health = mixed.read_stable_health(port, deadline, label)
        batching = mixed.batching_snapshot(health)
        graph = mixed.graph_snapshot(health)
        prefix = prefix_snapshot(health)
        last = {"batching": batching, "graph": graph, "prefix": prefix}
        if (
            mixed.batching_engine_drained(
                health["decode_runtime"]["batching_engine"]
            )
            and graph["active_graph_slot_count"] == 0
            and graph["tracked_decode_owner_count"] == 0
            and prefix["active_leases"] == 0
            and prefix["pending_release_entries"] == 0
            and batching["blocks_used"] == prefix["cached_blocks"]
        ):
            return health
        time.sleep(0.05)
    raise TimeoutError(f"{label} did not drain: {last!r}")


def canonical_hash(records: tuple[CompletionRecord, ...]) -> str:
    payload = [record.semantic for record in records]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def run_mode(
    binary: Path,
    model_path: Path,
    base_seed: int,
    mode: str,
    deadline: float,
) -> ModeRun:
    port = mixed.free_loopback_port()
    run_dir = ROOT / ".qualification/serving" / f"graph-correctness-{mode}-{os.getpid()}"
    adapter_dir = run_dir / "adapters"
    snapshot_dir = run_dir / "model-snapshots"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    environment = mixed.server_environment(
        mode, model_path, port, adapter_dir, snapshot_dir
    )
    environment.update(
        {
            "KILN_DETERMINISTIC": "1",
            "KILN_ROCM_GRAPH_CACHE_MAX": str(GRAPH_CACHE_MAX),
        }
    )
    process = subprocess.Popen(
        [str(binary), "--config", "/dev/null", "serve", "--served-model-id", mixed.MODEL_ID],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    server_log = mixed.ServerLog(process.stdout)
    server_log.start()
    shutdown: mixed.ShutdownOutcome | None = None
    residue: tuple[str, ...] = ()
    try:
        mixed.wait_ready(port, process, server_log, deadline)
        health = wait_drained(port, deadline, f"{mode} startup")
        debug = mixed.json_request(port, "GET", "/v1/debug/model-state")
        attestation_failures = mixed.attest_runtime(mode, health, debug)
        if attestation_failures:
            raise CorrectnessError("; ".join(attestation_failures))

        warm: list[CompletionRecord] = []
        for index, (name, words) in enumerate(SCENARIOS):
            seed = base_seed + index
            prompt = mixed.deterministic_prompt(
                mixed.workload_marker(base_seed, f"graph-correctness-{name}"), words
            )
            warm.append(completion_request(port, name, prompt, seed))
            wait_drained(port, deadline, f"{mode} warm {name}")
        health_before = wait_drained(port, deadline, f"{mode} warm completion")
        graph_before = mixed.graph_snapshot(health_before)
        prefix_before = prefix_snapshot(health_before)

        measured: list[CompletionRecord] = []
        for index, (name, words) in enumerate(SCENARIOS):
            seed = base_seed + index
            prompt = mixed.deterministic_prompt(
                mixed.workload_marker(base_seed, f"graph-correctness-{name}"), words
            )
            measured.append(completion_request(port, name, prompt, seed))
            wait_drained(port, deadline, f"{mode} measured {name}")
        health_after = wait_drained(port, deadline, f"{mode} measured completion")
        graph_after = mixed.graph_snapshot(health_after)
        prefix_after = prefix_snapshot(health_after)
        if process.poll() is not None:
            raise CorrectnessError(f"{mode} server exited before qualification drain")
        return_value = (
            tuple(warm),
            tuple(measured),
            graph_before,
            graph_after,
            prefix_after["lookup_hits"] - prefix_before["lookup_hits"],
        )
    finally:
        shutdown = mixed.terminate_process(process)
        server_log.join()
        residue = tuple(mixed.snapshot_payload_residue(snapshot_dir))
        shutil.rmtree(run_dir, ignore_errors=True)
    return ModeRun(
        mode=mode,
        warm=return_value[0],
        measured=return_value[1],
        graph_before_measure=return_value[2],
        graph_after_measure=return_value[3],
        prefix_hits=return_value[4],
        shutdown=shutdown,
        snapshot_residue=residue,
    )


def mismatch_counts(
    eager: tuple[CompletionRecord, ...], graph: tuple[CompletionRecord, ...]
) -> dict[str, int]:
    if len(eager) != len(graph):
        raise CorrectnessError("comparison record counts differ")
    output = 0
    token_ids = 0
    logprobs = 0
    for left, right in zip(eager, graph, strict=True):
        if left.scenario != right.scenario:
            raise CorrectnessError("comparison scenario order differs")
        if left.semantic != right.semantic:
            output += 1
        left_ids = tuple(action[1] for action in left.action_tokens)
        right_ids = tuple(action[1] for action in right.action_tokens)
        token_ids += int(left_ids != right_ids)
        logprobs += int(left.sampled_logprobs != right.sampled_logprobs)
    return {
        "output_mismatch_count": output,
        "token_id_mismatch_count": token_ids,
        "behavior_logprob_mismatch_count": logprobs,
    }


def first_mismatch(
    left: tuple[CompletionRecord, ...], right: tuple[CompletionRecord, ...]
) -> dict[str, Any] | None:
    if len(left) != len(right):
        return {"left_records": len(left), "right_records": len(right)}
    for left_record, right_record in zip(left, right, strict=True):
        if left_record.scenario != right_record.scenario:
            return {
                "left_scenario": left_record.scenario,
                "right_scenario": right_record.scenario,
            }
        for index, (left_action, right_action) in enumerate(
            zip(left_record.action_tokens, right_record.action_tokens)
        ):
            if left_action != right_action:
                return {
                    "action_index": index,
                    "left_action": list(left_action),
                    "right_action": list(right_action),
                    "scenario": left_record.scenario,
                }
        if len(left_record.action_tokens) != len(right_record.action_tokens):
            return {
                "left_action_count": len(left_record.action_tokens),
                "right_action_count": len(right_record.action_tokens),
                "scenario": left_record.scenario,
            }
        if left_record.semantic != right_record.semantic:
            return {
                "left_semantic_sha256": canonical_hash((left_record,)),
                "right_semantic_sha256": canonical_hash((right_record,)),
                "scenario": left_record.scenario,
            }
    return None


def metrics_from_values(values: dict[str, int | float]) -> list[dict[str, Any]]:
    if set(values) != set(METRIC_DEFINITIONS):
        missing = sorted(set(METRIC_DEFINITIONS) - set(values))
        extra = sorted(set(values) - set(METRIC_DEFINITIONS))
        raise CorrectnessError(f"metric set mismatch: missing={missing}, extra={extra}")
    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise CorrectnessError(f"metric {name} must be finite and nonnegative")
    return [
        {
            "name": name,
            "value": values[name],
            "unit": unit,
            "aggregation": aggregation,
            "lower_is_better": lower_is_better,
        }
        for name, (unit, aggregation, lower_is_better) in sorted(
            METRIC_DEFINITIONS.items()
        )
    ]


def execute(model_path: Path, seed: int) -> tuple[list[dict[str, Any]], str | None]:
    started = time.monotonic()
    deadline = started + OVERALL_TIMEOUT_SECONDS
    values: dict[str, int | float] = {name: 0 for name in METRIC_DEFINITIONS}
    values["scenario_count"] = len(SCENARIOS)
    failures: list[str] = []
    eager: ModeRun | None = None
    graph: ModeRun | None = None
    trace_hashes: dict[str, str] = {}
    mismatch_evidence: dict[str, dict[str, Any]] = {}
    try:
        binary, binary_hash, build_seconds = mixed.build_binary(deadline)
        mixed.trace(
            "graph_correctness_binary_built",
            build_seconds=build_seconds,
            path=str(binary.relative_to(ROOT)),
            sha256=binary_hash,
        )
        eager = run_mode(binary, model_path, seed, "eager", deadline)
        graph = run_mode(binary, model_path, seed, "graph", deadline)
        values["eager_request_count"] = len(eager.warm) + len(eager.measured)
        values["graph_request_count"] = len(graph.warm) + len(graph.measured)
        values["action_token_count"] = sum(
            len(record.action_tokens) for record in eager.measured
        )
        values["non_finite_logprob_count"] = sum(
            not math.isfinite(logprob)
            for mode_run in (eager, graph)
            for record in (*mode_run.warm, *mode_run.measured)
            for logprob in record.sampled_logprobs
        )
        eager_repeat = mismatch_counts(eager.warm, eager.measured)
        graph_repeat = mismatch_counts(graph.warm, graph.measured)
        cross = mismatch_counts(eager.measured, graph.measured)
        for label, left, right in (
            ("eager_repeat", eager.warm, eager.measured),
            ("graph_repeat", graph.warm, graph.measured),
            ("eager_vs_graph", eager.measured, graph.measured),
        ):
            if mismatch := first_mismatch(left, right):
                mismatch_evidence[label] = mismatch
        values["eager_repeat_mismatch_count"] = sum(eager_repeat.values())
        values["graph_repeat_mismatch_count"] = sum(graph_repeat.values())
        values.update(cross)

        eager_after = eager.graph_after_measure
        graph_before = graph.graph_before_measure
        graph_after = graph.graph_after_measure
        values["eager_graph_activity_count"] = sum(
            eager_after[field]
            for field in (
                "capture_attempts",
                "capture_successes",
                "replay_attempts",
                "replay_successes",
                "fallback_total",
            )
        )
        values["graph_capture_warm_count"] = graph_before["capture_successes"]
        values["graph_measure_capture_count"] = mixed.counter_delta(
            graph_before, graph_after, "capture_successes"
        )
        values["graph_measure_replay_count"] = mixed.counter_delta(
            graph_before, graph_after, "replay_successes"
        )
        values["graph_capture_failure_count"] = graph_after["capture_failures"]
        values["graph_replay_failure_count"] = graph_after["replay_failures"]
        values["graph_fallback_count"] = graph_after["fallback_total"]
        values["graph_retained_count_end"] = graph_after["captured_graph_count"]
        values["graph_slot_count_end"] = graph_after["graph_slot_count"]
        values["graph_slot_active_count_end"] = graph_after[
            "active_graph_slot_count"
        ]
        values["graph_slot_idle_count_end"] = graph_after["idle_graph_slot_count"]
        values["graph_slot_reuse_count"] = mixed.counter_delta(
            graph_before, graph_after, "graph_slot_reuse_count"
        )
        values["prefix_cache_hit_count"] = eager.prefix_hits + graph.prefix_hits

        for mode_run in (eager, graph):
            values["shutdown_forced_count"] += int(mode_run.shutdown.forced)
            values["shutdown_nonzero_count"] += int(mode_run.shutdown.returncode != 0)
            values["snapshot_residue_count"] += len(mode_run.snapshot_residue)
        trace_hashes = {
            "eager": canonical_hash(eager.measured),
            "graph": canonical_hash(graph.measured),
        }

        for name in (
            "behavior_logprob_mismatch_count",
            "eager_graph_activity_count",
            "eager_repeat_mismatch_count",
            "graph_capture_failure_count",
            "graph_fallback_count",
            "graph_measure_capture_count",
            "graph_replay_failure_count",
            "graph_repeat_mismatch_count",
            "graph_slot_active_count_end",
            "non_finite_logprob_count",
            "output_mismatch_count",
            "request_failure_count",
            "shutdown_forced_count",
            "shutdown_nonzero_count",
            "snapshot_residue_count",
            "token_id_mismatch_count",
        ):
            if values[name] != 0:
                failures.append(f"{name}={values[name]}, expected 0")
        if values["graph_capture_warm_count"] < 1:
            failures.append("graph warmup did not capture a supported prompt geometry")
        expected_decode_steps = values["action_token_count"] - len(SCENARIOS)
        if values["graph_measure_replay_count"] < expected_decode_steps:
            failures.append("measured graph replay count was below expected decode steps")
        if values["graph_retained_count_end"] > GRAPH_CACHE_MAX:
            failures.append("retained graph count exceeded the configured bound")
        if values["graph_slot_count_end"] > GRAPH_CACHE_MAX:
            failures.append("retained graph-slot count exceeded the configured bound")
        if values["graph_slot_idle_count_end"] != values["graph_slot_count_end"]:
            failures.append("not every graph slot was idle after measured drain")
        if values["graph_slot_reuse_count"] < len(SCENARIOS):
            failures.append("measured requests did not reuse graph slots")
        if values["prefix_cache_hit_count"] < len(SCENARIOS) * 2:
            failures.append("measured requests did not reuse every warmed prefix")
        if trace_hashes.get("eager") != trace_hashes.get("graph"):
            failures.append("canonical eager and graph trace hashes differ")
    except Exception as exc:
        values["request_failure_count"] = 1
        failures.append(f"{type(exc).__name__}: {exc}")

    if failures:
        details = " | ".join(dict.fromkeys(failures))
        if mismatch_evidence:
            details += " | mismatch_evidence=" + json.dumps(
                mismatch_evidence, sort_keys=True, separators=(",", ":")
            )
    else:
        details = json.dumps(
            {
                "action_tokens": values["action_token_count"],
                "eager_trace_sha256": trace_hashes["eager"],
                "graph_trace_sha256": trace_hashes["graph"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    return metrics_from_values(values), mixed.bounded_details(details)


def write_result(path: Path, metrics: list[dict[str, Any]], details: str | None, elapsed: float) -> None:
    passed = details is not None and details.startswith("{")
    value = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "status": "passed" if passed else "failed",
        "duration_seconds": elapsed,
        "effective_config": EFFECTIVE_CONFIG,
        "metrics": metrics,
        "tolerances": [],
        "details": details,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not args.model_path.is_dir():
        parser.error("--model-path must be a directory")
    return args


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    variant = os.environ.get(VARIANT_ENV, "")
    result_value = os.environ.get(RESULT_ENV)
    if variant != VARIANT_ID:
        print(f"{VARIANT_ENV} must be {VARIANT_ID!r}, got {variant!r}", file=os.sys.stderr)
        return 2
    if not result_value:
        print(f"{RESULT_ENV} is required", file=os.sys.stderr)
        return 2
    metrics, details = execute(args.model_path.resolve(), args.seed)
    status = 0 if details is not None and details.startswith("{") else 1
    write_result(Path(result_value), metrics, details, time.monotonic() - started)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
