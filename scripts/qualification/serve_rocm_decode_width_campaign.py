#!/usr/bin/env python3
"""Select a correctness-qualified ROCm decode width with bounded local trials."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import serve_mixed_load as mixed


CASE_ID = "rocm-decode-width-campaign"
RESULT_ENV = "KILN_QUALIFICATION_CASE_RESULT"
VARIANT_ENV = "KILN_QUALIFICATION_VARIANT_ID"
VARIANT_ID = "rocm"
CANDIDATE_WIDTHS = (2, 4, 8)
BASELINE_WIDTH = CANDIDATE_WIDTHS[0]
THROUGHPUT_TIE_PERCENT = 2.0
MIN_PROMOTION_IMPROVEMENT_PERCENT = 3.0
MAX_TAIL_REGRESSION_PERCENT = 25.0
MAX_PEAK_GPU_MEMORY_GROWTH_BYTES = 1 << 30

SOURCE_METRICS: tuple[tuple[str, str], ...] = (
    ("deterministic_output_tokens_per_second", "output_token_throughput_per_second"),
    ("sampled_output_tokens_per_second", "sampled_profile_output_token_throughput_per_second"),
    ("itl_ms_p99", "itl_ms_p99"),
    ("ttft_ms_p99", "ttft_ms_p99"),
    ("e2e_ms_p99", "e2e_latency_ms_p99"),
    ("max_observed_decode_batch", "batching_max_observed_batch_size"),
    (
        "sampled_max_observed_decode_batch",
        "sampled_profile_batching_max_observed_batch_size_end",
    ),
    ("graph_capture_success_count", "graph_measured_capture_success_count"),
    ("graph_capture_failure_count", "graph_measured_capture_failure_count"),
    ("graph_replay_success_count", "graph_measured_replay_success_count"),
    ("graph_replay_failure_count", "graph_measured_replay_failure_count"),
    (
        "sampled_fused_dispatch_count",
        "sampled_profile_rocm_w8_lm_head_sample_dispatch_count",
    ),
    (
        "sampled_fused_row_count",
        "sampled_profile_rocm_w8_lm_head_sample_row_count",
    ),
    (
        "sampled_fused_max_batch_rows",
        "sampled_profile_rocm_w8_lm_head_max_batch_rows_end",
    ),
    (
        "sampled_fused_failure_count",
        "sampled_profile_rocm_w8_lm_head_dispatch_failure_count",
    ),
    ("peak_gpu_memory_used_bytes", "peak_gpu_memory_used_bytes"),
    ("host_temperature_peak_millicelsius", "host_temperature_peak_millicelsius"),
    ("request_failure_count", "request_failure_count"),
    ("unexplained_itl_outlier_count", "unexplained_itl_outlier_count"),
    ("external_yield_sync_failure_count", "external_yield_sync_failure_count"),
)

SUMMARY_METRICS: tuple[tuple[str, str, str, bool], ...] = (
    ("candidate_count", "count", "exact", False),
    ("candidate_pass_count", "count", "sum", False),
    ("candidate_failure_count", "count", "sum", True),
    ("candidate_not_run_count", "count", "sum", True),
    ("selected_decode_width", "rows", "exact", False),
    ("selected_score_ratio", "ratio", "exact", False),
    (
        "selected_min_throughput_improvement_percent",
        "percent",
        "exact",
        False,
    ),
    ("source_build_seconds", "s", "exact", True),
)


class CampaignError(RuntimeError):
    """Raised when a width campaign cannot produce promotable evidence."""


@dataclass
class CandidateOutcome:
    width: int
    values: dict[str, float | int]
    correctness_reasons: list[str]
    performance_reasons: list[str]
    score_ratio: float = 0.0
    selected: bool = False
    not_run: bool = False

    @property
    def correctness_passed(self) -> bool:
        return not self.not_run and not self.correctness_reasons


def candidate_variant_id(width: int) -> str:
    return f"decode-width-{width}"


def candidate_config(width: int) -> dict[str, Any]:
    if width not in CANDIDATE_WIDTHS:
        raise CampaignError(f"undeclared decode-width candidate {width}")
    config = mixed._variant_config(
        serving_profile="experimental",
        kv_autoscale_requested=False,
        kv_autoscale_enabled=False,
        memory_reclaim_requested_mode="off",
        memory_reclaim_mode="off",
        rocm_graphs_requested=True,
        rocm_graphs_enabled=True,
        max_decode_batch=width,
        max_prefill_layers_per_cycle=mixed.MAX_PREFILL_LAYERS_PER_CYCLE,
        actor_cycle_idle_ms=mixed.ACTOR_CYCLE_IDLE_MS,
    )
    config["runtime"].update(
        {
            "prefix_cache_effective_enabled": True,
            "prefix_cache_effective_reason": "active",
            "prefix_cache_requested_enabled": True,
        }
    )
    return mixed._mixed_load_host_safety(config)


def effective_config() -> dict[str, Any]:
    shared = copy.deepcopy(candidate_config(BASELINE_WIDTH))
    server = shared["server"]
    for field in (
        "max_active_requests",
        "max_decode_batch",
        "max_prefill_staging_priority_burst",
        "max_prefill_staging_slots",
    ):
        del server[field]
    return {
        "build": shared.pop("build"),
        "candidate_policy": {
            "baseline_width": BASELINE_WIDTH,
            "candidate_widths": {
                f"width_{width}": width for width in CANDIDATE_WIDTHS
            },
            "max_active_requests_formula": "decode_width + prefill_staging_slots",
            "prefill_staging_slots_formula": (
                "min(decode_width, 4) when decode_width > 1, otherwise 0"
            ),
        },
        "selection": {
            "max_peak_gpu_memory_growth_bytes": MAX_PEAK_GPU_MEMORY_GROWTH_BYTES,
            "max_tail_regression_percent": MAX_TAIL_REGRESSION_PERCENT,
            "min_promotion_improvement_percent": (
                MIN_PROMOTION_IMPROVEMENT_PERCENT
            ),
            "objective": (
                "maximize the minimum deterministic/sample throughput ratio "
                "to width 2, then choose the narrowest width within the tie band"
            ),
            "throughput_tie_percent": THROUGHPUT_TIE_PERCENT,
        },
        "shared_candidate_config": shared,
    }


EFFECTIVE_CONFIG = effective_config()


def register_candidate_configs() -> None:
    for width in CANDIDATE_WIDTHS:
        mixed.VARIANT_CONFIGS[candidate_variant_id(width)] = candidate_config(width)


def metric_map(metrics: list[dict[str, Any]]) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    for metric in metrics:
        name = metric.get("name")
        value = metric.get("value")
        if not isinstance(name, str):
            raise CampaignError("candidate metric omitted its name")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise CampaignError(f"candidate metric {name!r} is not finite numeric evidence")
        if name in values:
            raise CampaignError(f"candidate metric {name!r} is duplicated")
        values[name] = value
    missing = sorted(source for _, source in SOURCE_METRICS if source not in values)
    if missing:
        raise CampaignError(f"candidate metrics are missing {missing}")
    return values


def correctness_reasons(
    width: int,
    values: dict[str, float | int],
    execution_details: str | None,
) -> list[str]:
    reasons = [execution_details] if execution_details else []
    exact_width_fields = (
        "batching_max_decode_batch",
        "batching_max_observed_batch_size",
        "sampled_profile_batching_max_observed_batch_size_end",
        "sampled_profile_rocm_w8_lm_head_max_batch_rows_end",
    )
    for field in exact_width_fields:
        observed = values.get(field)
        if observed != width:
            reasons.append(f"{field}={observed!r}, expected {width}")
    zero_fields = (
        "external_yield_sync_failure_count",
        "graph_measured_capture_failure_count",
        "graph_measured_replay_failure_count",
        "request_failure_count",
        "sampled_profile_request_failure_count",
        "sampled_profile_rocm_w8_lm_head_dispatch_failure_count",
        "unexplained_itl_outlier_count",
        "zero_token_response_count",
    )
    for field in zero_fields:
        if values.get(field) != 0:
            reasons.append(f"{field}={values.get(field)!r}, expected 0")
    positive_fields = (
        "graph_measured_replay_success_count",
        "sampled_profile_rocm_w8_lm_head_sample_dispatch_count",
        "sampled_profile_rocm_w8_lm_head_sample_row_count",
    )
    for field in positive_fields:
        value = values.get(field)
        if not isinstance(value, (int, float)) or value <= 0:
            reasons.append(f"{field}={value!r}, expected positive evidence")
    capture_successes = values.get("graph_pre_measurement_capture_success_count", 0)
    capture_successes += values.get("graph_measured_capture_success_count", 0)
    if capture_successes <= 0:
        reasons.append("graph capture recorded no successful warmup or measured event")
    return reasons


def _regression_percent(candidate: float | int, baseline: float | int) -> float:
    if baseline <= 0:
        raise CampaignError("baseline latency or memory metric must be positive")
    return (float(candidate) / float(baseline) - 1.0) * 100.0


def select_candidate(outcomes: list[CandidateOutcome]) -> CandidateOutcome:
    by_width = {outcome.width: outcome for outcome in outcomes}
    baseline = by_width.get(BASELINE_WIDTH)
    if baseline is None or not baseline.correctness_passed:
        raise CampaignError("the baseline decode width did not pass correctness")
    if any(not outcome.correctness_passed for outcome in outcomes):
        raise CampaignError("not every declared decode-width candidate passed correctness")

    baseline_deterministic = baseline.values["output_token_throughput_per_second"]
    baseline_sampled = baseline.values[
        "sampled_profile_output_token_throughput_per_second"
    ]
    if baseline_deterministic <= 0 or baseline_sampled <= 0:
        raise CampaignError("baseline throughput must be positive")

    eligible: list[CandidateOutcome] = []
    for outcome in outcomes:
        deterministic_ratio = (
            float(outcome.values["output_token_throughput_per_second"])
            / float(baseline_deterministic)
        )
        sampled_ratio = (
            float(outcome.values["sampled_profile_output_token_throughput_per_second"])
            / float(baseline_sampled)
        )
        outcome.score_ratio = min(deterministic_ratio, sampled_ratio)
        if outcome.width != BASELINE_WIDTH:
            for field in ("itl_ms_p99", "ttft_ms_p99", "e2e_latency_ms_p99"):
                regression = _regression_percent(
                    outcome.values[field], baseline.values[field]
                )
                if regression > MAX_TAIL_REGRESSION_PERCENT:
                    outcome.performance_reasons.append(
                        f"{field} regressed {regression:.2f}%"
                    )
            memory_growth = int(outcome.values["peak_gpu_memory_used_bytes"]) - int(
                baseline.values["peak_gpu_memory_used_bytes"]
            )
            if memory_growth > MAX_PEAK_GPU_MEMORY_GROWTH_BYTES:
                outcome.performance_reasons.append(
                    f"peak GPU memory grew {memory_growth} bytes"
                )
        if not outcome.performance_reasons:
            eligible.append(outcome)

    best_score = max(outcome.score_ratio for outcome in eligible)
    tie_floor = best_score * (1.0 - THROUGHPUT_TIE_PERCENT / 100.0)
    selected = min(
        (outcome for outcome in eligible if outcome.score_ratio >= tie_floor),
        key=lambda outcome: outcome.width,
    )
    promotion_floor = 1.0 + MIN_PROMOTION_IMPROVEMENT_PERCENT / 100.0
    if selected.width != BASELINE_WIDTH and selected.score_ratio < promotion_floor:
        selected.performance_reasons.append(
            "gain did not clear the minimum promotion threshold"
        )
        selected = baseline
    selected.selected = True
    return selected


def result_metric(
    name: str,
    value: float | int,
    unit: str,
    aggregation: str,
    lower_is_better: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "aggregation": aggregation,
        "lower_is_better": lower_is_better,
    }


def declared_metric_names() -> list[str]:
    names = [name for name, _, _, _ in SUMMARY_METRICS]
    for width in CANDIDATE_WIDTHS:
        names.append(f"width_{width}_correctness_passed")
        names.append(f"width_{width}_score_ratio")
        for target, _ in SOURCE_METRICS:
            names.append(f"width_{width}_{target}")
    return sorted(names)


def result_metrics(
    outcomes: list[CandidateOutcome],
    selected: CandidateOutcome | None,
    build_seconds: float,
) -> list[dict[str, Any]]:
    by_width = {outcome.width: outcome for outcome in outcomes}
    passed = sum(outcome.correctness_passed for outcome in outcomes)
    not_run = sum(outcome.not_run for outcome in outcomes)
    summary_values: dict[str, float | int] = {
        "candidate_count": len(CANDIDATE_WIDTHS),
        "candidate_pass_count": passed,
        "candidate_failure_count": len(CANDIDATE_WIDTHS) - passed - not_run,
        "candidate_not_run_count": not_run,
        "selected_decode_width": selected.width if selected is not None else 0,
        "selected_score_ratio": selected.score_ratio if selected is not None else 0.0,
        "selected_min_throughput_improvement_percent": (
            (selected.score_ratio - 1.0) * 100.0 if selected is not None else 0.0
        ),
        "source_build_seconds": build_seconds,
    }
    metrics = [
        result_metric(name, summary_values[name], unit, aggregation, lower)
        for name, unit, aggregation, lower in SUMMARY_METRICS
    ]
    for width in CANDIDATE_WIDTHS:
        outcome = by_width.get(width)
        metrics.append(
            result_metric(
                f"width_{width}_correctness_passed",
                int(outcome.correctness_passed) if outcome is not None else 0,
                "bool",
                "exact",
                False,
            )
        )
        metrics.append(
            result_metric(
                f"width_{width}_score_ratio",
                outcome.score_ratio if outcome is not None else 0.0,
                "ratio",
                "exact",
                False,
            )
        )
        for target, source in SOURCE_METRICS:
            definition = mixed.METRIC_DEFINITIONS[source]
            value = outcome.values.get(source, 0) if outcome is not None else 0
            metrics.append(
                result_metric(
                    f"width_{width}_{target}", value, *definition
                )
            )
    return sorted(metrics, key=lambda metric: metric["name"])


def compact_summary(
    outcomes: list[CandidateOutcome], selected: CandidateOutcome | None
) -> str:
    rows = []
    for outcome in outcomes:
        reasons = [*outcome.correctness_reasons, *outcome.performance_reasons]
        rows.append(
            {
                "width": outcome.width,
                "verdict": (
                    "not_run"
                    if outcome.not_run
                    else "selected"
                    if outcome.selected
                    else "rejected"
                    if reasons
                    else "qualified_not_selected"
                ),
                "score_ratio": round(outcome.score_ratio, 6),
                "deterministic_tok_s": round(
                    float(outcome.values.get("output_token_throughput_per_second", 0)),
                    6,
                ),
                "sampled_tok_s": round(
                    float(
                        outcome.values.get(
                            "sampled_profile_output_token_throughput_per_second", 0
                        )
                    ),
                    6,
                ),
                "reasons": reasons[:3],
            }
        )
    return mixed.bounded_details(
        json.dumps(
            {"selected_width": selected.width if selected else None, "candidates": rows},
            sort_keys=True,
            separators=(",", ":"),
        )
    ) or ""


def zero_outcome(width: int, *, not_run: bool, reason: str) -> CandidateOutcome:
    return CandidateOutcome(
        width=width,
        values={source: 0 for _, source in SOURCE_METRICS},
        correctness_reasons=[reason],
        performance_reasons=[],
        not_run=not_run,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args(argv)


def run_campaign(
    model_path: Path, seed: int
) -> tuple[list[CandidateOutcome], CandidateOutcome, float]:
    register_candidate_configs()
    build_started = time.monotonic()
    binary, binary_hash, _ = mixed.build_binary(
        build_started + mixed.BUILD_TIMEOUT_SECONDS
    )
    build_seconds = time.monotonic() - build_started
    outcomes: list[CandidateOutcome] = []
    for index, width in enumerate(CANDIDATE_WIDTHS):
        metrics, execution_details = mixed.execute(
            model_path,
            seed,
            candidate_variant_id(width),
            built_binary=(binary, binary_hash),
        )
        values = metric_map(metrics)
        reasons = correctness_reasons(width, values, execution_details)
        outcomes.append(
            CandidateOutcome(
                width=width,
                values=values,
                correctness_reasons=reasons,
                performance_reasons=[],
            )
        )
        if reasons:
            for skipped_width in CANDIDATE_WIDTHS[index + 1 :]:
                outcomes.append(
                    zero_outcome(
                        skipped_width,
                        not_run=True,
                        reason=f"not run after width {width} failed correctness",
                    )
                )
            break
    return outcomes, select_candidate(outcomes), build_seconds


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    result_path_value = os.environ.get(RESULT_ENV)
    if not result_path_value:
        print(f"{RESULT_ENV} is required", file=os.sys.stderr)
        return 2
    result_path = Path(result_path_value)
    outcomes: list[CandidateOutcome] = []
    selected: CandidateOutcome | None = None
    build_seconds = 0.0
    status = "failed"
    failure: str | None = None
    previous_sigterm = signal.signal(signal.SIGTERM, mixed.raise_termination_interrupt)
    try:
        variant = os.environ.get(VARIANT_ENV, "")
        if variant != VARIANT_ID:
            raise CampaignError(f"{VARIANT_ENV} must be {VARIANT_ID!r}, got {variant!r}")
        model_path = args.model_path.resolve(strict=True)
        if not model_path.is_dir():
            raise CampaignError("--model-path must be a directory")
        outcomes, selected, build_seconds = run_campaign(model_path, args.seed)
        status = "passed"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        mixed.trace("decode_width_campaign_error", details=failure)
        known_widths = {outcome.width for outcome in outcomes}
        for width in CANDIDATE_WIDTHS:
            if width not in known_widths:
                outcomes.append(
                    zero_outcome(width, not_run=True, reason="campaign stopped before arm")
                )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    summary = compact_summary(outcomes, selected)
    details = summary if failure is None else mixed.bounded_details(f"{failure} | {summary}")
    result = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "status": status,
        "duration_seconds": time.monotonic() - started,
        "effective_config": EFFECTIVE_CONFIG,
        "metrics": result_metrics(outcomes, selected, build_seconds),
        "tolerances": [],
        "details": details,
    }
    try:
        mixed.write_result(result_path, result)
    except Exception as exc:
        print(f"cannot write qualification result: {exc}", file=os.sys.stderr)
        return 2
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
