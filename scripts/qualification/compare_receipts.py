#!/usr/bin/env python3
"""Compare qualification receipts under their committed workload policy."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, DecimalException, localcontext
from pathlib import Path
from typing import Any

from receipt import validate_receipt
from workload import (
    WorkloadLoadError,
    WorkloadValidationError,
    load_workload_document,
    runner_metric_definition,
    validate_workload,
    validate_workload_parameters,
    workload_file_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_DIRECTORY = Path("qualification/workloads")
JSON_INTEGER_MAX_DIGITS = 4096


class ComparisonError(RuntimeError):
    """Raised when an input cannot be trusted enough to compare."""


@dataclass(frozen=True)
class LoadedReceipt:
    path: Path
    value: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class LoadedWorkload:
    root: Path
    path: Path
    value: dict[str, Any]
    sha256: str


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _strict_json_object(payload: bytes, path: Path, kind: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ComparisonError(f"non-finite JSON number is not allowed: {value}")

    def parse_finite_float(value: str) -> float:
        try:
            exact = Decimal(value)
            parsed = float(value)
        except (DecimalException, OverflowError, ValueError) as exc:
            raise ComparisonError(f"invalid JSON number: {value}") from exc
        if not math.isfinite(parsed):
            raise ComparisonError(f"JSON number overflows finite float range: {value}")
        if parsed == 0.0:
            if exact != 0:
                raise ComparisonError(f"JSON number underflows finite float range: {value}")
            return 0.0
        if Decimal(str(parsed)) != exact:
            raise ComparisonError(f"JSON number is not exactly representable: {value}")
        return parsed

    def parse_bounded_int(value: str) -> int:
        if len(value.lstrip("-")) > JSON_INTEGER_MAX_DIGITS:
            raise ComparisonError(
                f"JSON integer exceeds {JSON_INTEGER_MAX_DIGITS} digits"
            )
        try:
            return int(value)
        except ValueError as exc:
            raise ComparisonError(f"invalid JSON integer: {value}") from exc

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ComparisonError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
            parse_float=parse_finite_float,
            parse_int=parse_bounded_int,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ComparisonError) as exc:
        raise ComparisonError(f"cannot load {kind} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"{kind} {path} must be a JSON object")
    return value


def load_validated_receipt(path: Path) -> LoadedReceipt:
    """Load and validate a receipt from the same bytes used for its identity."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ComparisonError(f"cannot read receipt {path}: {exc}") from exc
    value = _strict_json_object(payload, path, "receipt")
    errors = validate_receipt(value)
    if errors:
        detail = "\n".join(f"  - {error}" for error in errors)
        raise ComparisonError(f"receipt {path} failed validation:\n{detail}")
    return LoadedReceipt(path=path, value=value, sha256=_sha256(payload))


def _committed_bytes(root: Path, relative_path: Path) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "show", f"HEAD:{relative_path.as_posix()}"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode(errors="replace").strip()
        raise ComparisonError(
            f"workload manifest must be committed at HEAD: {relative_path}"
            + (f" ({detail})" if detail else "")
        ) from exc
    return completed.stdout


def load_committed_workload(path: Path, *, root: Path) -> LoadedWorkload:
    """Load a valid workload whose exact bytes match the repository HEAD."""
    root = root.resolve()
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root)
        relative.relative_to(WORKLOAD_DIRECTORY)
    except (OSError, ValueError) as exc:
        raise ComparisonError(
            f"workload manifest must resolve under {root / WORKLOAD_DIRECTORY}"
        ) from exc
    if resolved.suffix != ".json":
        raise ComparisonError("workload manifest must be a JSON file")

    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise ComparisonError(f"cannot read workload manifest {resolved}: {exc}") from exc
    if payload != _committed_bytes(root, relative):
        raise ComparisonError(
            f"workload manifest must exactly match its committed HEAD version: {relative}"
        )

    try:
        workload, loaded_payload = load_workload_document(resolved)
        errors = validate_workload(workload)
        digest = workload_file_sha256(resolved) if not errors else ""
    except (WorkloadLoadError, WorkloadValidationError) as exc:
        raise ComparisonError(str(exc)) from exc
    if loaded_payload != payload or (digest and digest != _sha256(payload)):
        raise ComparisonError(f"workload manifest changed while it was being loaded: {relative}")
    if errors:
        detail = "\n".join(f"  - {error}" for error in errors)
        raise ComparisonError(f"workload manifest {relative} failed validation:\n{detail}")
    return LoadedWorkload(root=root, path=relative, value=workload, sha256=digest)


def _json_equal(baseline: Any, candidate: Any) -> bool:
    """Compare JSON without Python's bool/int/float equality coercions."""
    options = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": True}
    return json.dumps(baseline, **options) == json.dumps(candidate, **options)


def _verify_committed_workload(manifest: LoadedWorkload) -> None:
    """Prevent programmatic callers from injecting an uncommitted policy value."""
    current = load_committed_workload(manifest.root / manifest.path, root=manifest.root)
    if current.sha256 != manifest.sha256 or not _json_equal(current.value, manifest.value):
        raise ComparisonError("comparison policy must come from the exact committed workload")


def _model_identity(model: Any) -> Any:
    if model is None:
        return None
    # The absolute root is host placement. Shard paths within that root remain
    # content identity alongside every required content hash.
    return {
        "id": model["id"],
        "weight_files": sorted(model["weight_files"], key=lambda item: item["path"]),
        "config_hash": model["config_hash"],
        "tokenizer_hash": model["tokenizer_hash"],
        "chat_template_hash": model["chat_template_hash"],
    }


def _metric_definition(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": metric["name"],
        "unit": metric["unit"],
        "aggregation": metric["aggregation"],
        "lower_is_better": metric["lower_is_better"],
    }


def _metric_scope(scope: str, metric: dict[str, Any]) -> str:
    return f"{scope}/{metric['name']}"


def _metrics_by_scope(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for metric in receipt["metrics"]:
        metrics[_metric_scope("receipt.metrics", metric)] = metric
    for result in receipt["results"]:
        scope = f"results/{result['id']}/metrics"
        for metric in result["metrics"]:
            metrics[_metric_scope(scope, metric)] = metric
    return metrics


def _metric_identity(receipt: dict[str, Any]) -> dict[str, Any]:
    metrics = _metrics_by_scope(receipt)
    return {
        "results": [
            {"id": result["id"], "required": result["required"]}
            for result in sorted(receipt["results"], key=lambda item: item["id"])
        ],
        "metrics": {
            scope: _metric_definition(metrics[scope])
            for scope in sorted(metrics)
        },
    }


def _workload_base_identity(workload: dict[str, Any]) -> dict[str, Any]:
    parameters = {
        key: 0.0 if isinstance(value, float) and value == 0 else value
        for key, value in workload["parameters"].items()
        if key != "variant_id"
    }
    return {
        "id": workload["id"],
        "sha256": workload["sha256"],
        "seed": workload["seed"],
        "parameters": parameters,
    }


def _variants_by_id(manifest: LoadedWorkload) -> dict[str, dict[str, Any]]:
    return {variant["id"]: variant for variant in manifest.value["variants"]}


def _bind_manifest(
    baseline: LoadedReceipt,
    candidate: LoadedReceipt,
    manifest: LoadedWorkload,
) -> tuple[str, str]:
    variants = _variants_by_id(manifest)
    selected: list[str] = []
    for label, receipt in (("baseline", baseline), ("candidate", candidate)):
        workload = receipt.value["workload"]
        if not isinstance(workload, dict):
            raise ComparisonError(f"{label} receipt must name a workload manifest")
        if workload["id"] != manifest.value["workload_id"]:
            raise ComparisonError(
                f"{label} receipt workload id does not match committed manifest"
            )
        if workload["sha256"] != manifest.sha256:
            raise ComparisonError(
                f"{label} receipt workload sha256 does not match exact committed manifest bytes"
            )
        if not _json_equal(workload["seed"], manifest.value["determinism"]["seed"]):
            raise ComparisonError(f"{label} receipt seed does not match workload manifest")
        parameter_errors = validate_workload_parameters(
            manifest.value, workload["parameters"]
        )
        if parameter_errors:
            detail = "\n".join(f"  - {error}" for error in parameter_errors)
            raise ComparisonError(
                f"{label} receipt parameters do not match committed workload:\n{detail}"
            )
        variant_id = workload["parameters"].get("variant_id")
        if not isinstance(variant_id, str) or variant_id not in variants:
            raise ComparisonError(
                f"{label} receipt workload.parameters.variant_id must name a declared variant"
            )
        backend = receipt.value["qualification"]["backend"]
        if backend != variants[variant_id]["backend"]:
            raise ComparisonError(
                f"{label} receipt backend {backend!r} does not match variant "
                f"{variant_id!r} backend {variants[variant_id]['backend']!r}"
            )
        kind = receipt.value["qualification"]["kind"]
        if kind != manifest.value["kind"]:
            raise ComparisonError(
                f"{label} receipt kind {kind!r} does not match workload kind "
                f"{manifest.value['kind']!r}"
            )
        selected.append(variant_id)
    return selected[0], selected[1]


def _add_difference(
    differences: list[dict[str, Any]],
    compatibility: str,
    baseline: Any,
    candidate: Any,
) -> None:
    if not _json_equal(baseline, candidate):
        differences.append(
            {
                "compatibility": compatibility,
                "baseline": baseline,
                "candidate": candidate,
            }
        )


def _config_difference_paths(baseline: Any, candidate: Any, prefix: str = "") -> list[str]:
    if _json_equal(baseline, candidate):
        return []
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        differences: list[str] = []
        for key in sorted(baseline.keys() | candidate.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in baseline or key not in candidate:
                differences.append(path)
            else:
                differences.extend(
                    _config_difference_paths(baseline[key], candidate[key], path)
                )
        return differences
    return [prefix or "<root>"]


def _dot_value(value: dict[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = value
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return False, None
        current = current[segment]
    return True, current


def _allowed_environment_differences(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    allowed_paths: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    allowed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in allowed_paths:
        before_exists, before = _dot_value(baseline, path)
        after_exists, after = _dot_value(candidate, path)
        if not before_exists and not after_exists:
            errors.append(f"allowed environment path {path!r} is absent from both receipts")
        elif (before_exists and isinstance(before, (dict, list))) or (
            after_exists and isinstance(after, (dict, list))
        ):
            errors.append(
                f"allowed environment path {path!r} must name a scalar leaf"
            )
        elif before_exists and after_exists and _json_equal(before, after):
            errors.append(
                f"allowed environment path {path!r} is unused because its values are equal"
            )
        else:
            allowed.append(
                {
                    "compatibility": "environment",
                    "path": path,
                    "baseline": before if before_exists else None,
                    "candidate": after if after_exists else None,
                    "baseline_present": before_exists,
                    "candidate_present": after_exists,
                }
            )

    for path in _config_difference_paths(baseline, candidate):
        if path not in allowed_paths:
            rejected.append(
                {
                    "compatibility": "environment",
                    "path": path,
                    "baseline": baseline,
                    "candidate": candidate,
                }
            )
    return allowed, rejected, errors


def _backend_endpoint(backend: str, variant_id: str) -> tuple[str, str]:
    return backend, variant_id


def _matched_backend_pair(
    policy: dict[str, Any],
    baseline_backend: str,
    baseline_variant: str,
    candidate_backend: str,
    candidate_variant: str,
) -> dict[str, Any] | None:
    observed = {
        _backend_endpoint(baseline_backend, baseline_variant),
        _backend_endpoint(candidate_backend, candidate_variant),
    }
    for pair in policy["backend_pairs"]:
        declared = {
            _backend_endpoint(pair["backend_a"], pair["variant_a_id"]),
            _backend_endpoint(pair["backend_b"], pair["variant_b_id"]),
        }
        if observed == declared:
            return pair
    return None


def _required_result_errors(
    receipt: LoadedReceipt,
    variant: dict[str, Any],
    label: str,
    repetitions: int,
) -> list[str]:
    errors: list[str] = []
    results = {result["id"]: result for result in receipt.value["results"]}
    expected = {case["id"]: case["required"] for case in variant["cases"]}
    actual = {result_id: result["required"] for result_id, result in results.items()}
    if receipt.value["qualification"]["verdict"] != "passed":
        errors.append(f"{label} receipt verdict must be passed")
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    if missing:
        errors.append(f"{label} receipt is missing declared results: {', '.join(missing)}")
    if extra:
        errors.append(f"{label} receipt contains undeclared results: {', '.join(extra)}")
    for result_id in sorted(expected.keys() & actual.keys()):
        if actual[result_id] is not expected[result_id]:
            errors.append(
                f"{label} result {result_id!r} required flag does not match selected variant"
            )
    for case in variant["cases"]:
        result = results.get(case["id"])
        if case["required"] and result is not None and result["status"] != "passed":
            errors.append(f"{label} required result {case['id']!r} did not pass")
        protocol = case["result_protocol"]
        if result is not None:
            expected_metrics = set(protocol["declared_metrics"])
            actual_metrics = {metric["name"] for metric in result["metrics"]}
            if actual_metrics != expected_metrics:
                errors.append(
                    f"{label} result {case['id']!r} metric names do not match selected variant"
                )
        if result is None or protocol["producer"] != "runner":
            continue
        metrics = {metric["name"]: metric for metric in result["metrics"]}
        for name in protocol["declared_metrics"]:
            metric = metrics.get(name)
            if metric is None:
                continue
            expected_definition = runner_metric_definition(name, repetitions)
            assert expected_definition is not None
            observed_definition = {
                "unit": metric["unit"],
                "aggregation": metric["aggregation"],
                "lower_is_better": metric["lower_is_better"],
            }
            if observed_definition != expected_definition:
                errors.append(
                    f"{label} runner result {case['id']!r} metric {name!r} "
                    "has non-canonical definition"
                )

        case_pass = metrics.get("case_pass")
        if case_pass is None:
            continue
        expected_value = 1 if result["status"] == "passed" else 0
        value = case_pass["value"]
        try:
            valid_case_pass = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value == expected_value
            )
        except OverflowError:
            valid_case_pass = False
        if not valid_case_pass:
            errors.append(
                f"{label} runner result {case['id']!r} case_pass must be numeric "
                f"{expected_value} for status {result['status']!r}"
            )
        if result["status"] == "passed" and "exit_code" in metrics:
            exit_code = metrics["exit_code"]["value"]
            if (
                not isinstance(exit_code, int)
                or isinstance(exit_code, bool)
                or exit_code not in case["expected_exit_codes"]
            ):
                errors.append(
                    f"{label} passed runner result {case['id']!r} exit_code must be an "
                    "allowed integer agreed across all repetitions"
                )
        if result["status"] == "passed" and "output_assertion_failures" in metrics:
            failures = metrics["output_assertion_failures"]["value"]
            if isinstance(failures, bool) or failures != 0:
                errors.append(
                    f"{label} passed runner result {case['id']!r} "
                    "output_assertion_failures must be numeric 0"
                )
    return errors


def _metric_rule_scope(rule: dict[str, Any]) -> str:
    return f"results/{rule['result_id']}/metrics/{rule['metric']}"


def _finite_change(candidate: float | int, baseline: float | int) -> float | int | None:
    change = candidate - baseline
    try:
        return change if math.isfinite(change) else None
    except OverflowError:
        return None


def _evaluate_metric_rule(
    rule: dict[str, Any],
    baseline_metrics: dict[str, dict[str, Any]],
    candidate_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scope = _metric_rule_scope(rule)
    before = baseline_metrics.get(scope)
    after = candidate_metrics.get(scope)
    base = {
        "scope": scope,
        "metric_class": rule["metric_class"],
        "unit": rule["unit"],
        "aggregation": rule["aggregation"],
        "lower_is_better": rule["lower_is_better"],
        "operator": rule["operator"],
        "absolute_tolerance": rule["absolute_tolerance"],
        "relative_tolerance": rule["relative_tolerance"],
        "required": rule["required"],
    }
    if before is None or after is None:
        return {
            **base,
            "status": "failed" if rule["required"] else "skipped",
            "reason": "declared metric is missing from one or both receipts",
            "baseline": None if before is None else before["value"],
            "candidate": None if after is None else after["value"],
            "absolute_change": None,
            "relative_change_percent": None,
        }

    baseline_value = before["value"]
    candidate_value = after["value"]
    change = _finite_change(candidate_value, baseline_value)
    try:
        decimals = [
            Decimal(str(baseline_value)),
            Decimal(str(candidate_value)),
            Decimal(str(rule["absolute_tolerance"])),
            Decimal(str(rule["relative_tolerance"])),
        ]
        precision = max(
            80,
            sum(len(value.as_tuple().digits) for value in decimals)
            + max(abs(value.adjusted()) if value else 0 for value in decimals)
            + 32,
        )
        with localcontext() as context:
            context.prec = precision
            baseline_decimal, candidate_decimal, absolute_tolerance, relative_tolerance = decimals
            tolerance = max(
                absolute_tolerance,
                abs(baseline_decimal) * relative_tolerance,
            )
            if rule["operator"] == "equal":
                violation = abs(candidate_decimal - baseline_decimal)
            elif rule["operator"] == "not_greater":
                violation = max(Decimal(0), candidate_decimal - baseline_decimal)
            else:
                violation = max(Decimal(0), baseline_decimal - candidate_decimal)
            derived_finite = all(
                value.is_finite()
                for value in (baseline_decimal, candidate_decimal, tolerance, violation)
            )
            passed = derived_finite and violation <= tolerance
    except (DecimalException, ValueError):
        tolerance = Decimal("NaN")
        violation = Decimal("NaN")
        passed = False
    relative: float | None = None
    if baseline_value != 0 and change is not None:
        candidate_relative = (float(change) / abs(float(baseline_value))) * 100.0
        if math.isfinite(candidate_relative):
            relative = candidate_relative
    return {
        **base,
        "status": "passed" if passed else "failed",
        "reason": None
        if passed
        else f"violation {violation} exceeds tolerance {tolerance}",
        "baseline": baseline_value,
        "candidate": candidate_value,
        "absolute_change": change,
        "relative_change_percent": relative,
    }


def _metric_policy_errors(
    policy: dict[str, Any],
    baseline_metrics: dict[str, dict[str, Any]],
    candidate_metrics: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for rule in policy["metric_rules"]:
        scope = _metric_rule_scope(rule)
        expected = {
            "name": rule["metric"],
            "unit": rule["unit"],
            "aggregation": rule["aggregation"],
            "lower_is_better": rule["lower_is_better"],
        }
        for label, metrics in (("baseline", baseline_metrics), ("candidate", candidate_metrics)):
            metric = metrics.get(scope)
            if metric is not None and _metric_definition(metric) != expected:
                errors.append(
                    f"{label} metric {scope!r} definition does not match committed rule"
                )
    return errors


def compare_receipts(
    baseline: LoadedReceipt,
    candidate: LoadedReceipt,
    *,
    manifest: LoadedWorkload | None,
) -> dict[str, Any]:
    allowed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    compatibility_errors: list[str] = []
    evidence_errors: list[str] = []
    mode = "strict_no_workload"
    policy: dict[str, Any] | None = None

    if manifest is None:
        if baseline.value["workload"] is not None or candidate.value["workload"] is not None:
            raise ComparisonError(
                "--workload-manifest is required when either receipt names a workload"
            )
        if any(
            receipt.value["qualification"]["kind"] != "environment"
            for receipt in (baseline, candidate)
        ):
            raise ComparisonError(
                "a committed workload manifest is required for non-environment receipts"
            )
        baseline_variant = candidate_variant = ""
    else:
        _verify_committed_workload(manifest)
        baseline_variant, candidate_variant = _bind_manifest(baseline, candidate, manifest)
        if baseline.value["receipt_id"] == candidate.value["receipt_id"]:
            raise ComparisonError("manifest-bound comparison requires distinct receipt IDs")
        if baseline.sha256 == candidate.sha256:
            raise ComparisonError("manifest-bound comparison requires distinct receipt content")
        policy = manifest.value["comparison_policy"]
        if not isinstance(policy, dict):
            raise ComparisonError("the committed workload does not declare a comparison policy")
        mode = policy["mode"]

    baseline_value = baseline.value
    candidate_value = candidate.value
    _add_difference(
        rejected,
        "qualification_kind",
        baseline_value["qualification"]["kind"],
        candidate_value["qualification"]["kind"],
    )
    baseline_source = baseline_value["source"]
    candidate_source = candidate_value["source"]
    _add_difference(
        rejected,
        "source",
        {
            "tree_hash_format": baseline_source["tree_hash_format"],
            "tree_hash": baseline_source["tree_hash"],
        },
        {
            "tree_hash_format": candidate_source["tree_hash_format"],
            "tree_hash": candidate_source["tree_hash"],
        },
    )
    _add_difference(
        rejected,
        "model",
        _model_identity(baseline_value["model"]),
        _model_identity(candidate_value["model"]),
    )
    _add_difference(
        rejected,
        "profile",
        baseline_value["qualification"]["profile"],
        candidate_value["qualification"]["profile"],
    )
    _add_difference(
        rejected,
        "metric_identity",
        _metric_identity(baseline_value),
        _metric_identity(candidate_value),
    )

    if manifest is None:
        _add_difference(rejected, "workload", baseline_value["workload"], candidate_value["workload"])
        _add_difference(
            rejected,
            "backend_environment",
            {
                "backend": baseline_value["qualification"]["backend"],
                "environment": baseline_value["environment"],
            },
            {
                "backend": candidate_value["qualification"]["backend"],
                "environment": candidate_value["environment"],
            },
        )
        _add_difference(
            rejected,
            "effective_config",
            baseline_value["effective_config"],
            candidate_value["effective_config"],
        )
    else:
        baseline_workload = baseline_value["workload"]
        candidate_workload = candidate_value["workload"]
        assert isinstance(baseline_workload, dict) and isinstance(candidate_workload, dict)
        _add_difference(
            rejected,
            "workload",
            _workload_base_identity(baseline_workload),
            _workload_base_identity(candidate_workload),
        )
        variants = _variants_by_id(manifest)
        for label, observed, variant_id in (
            ("baseline", baseline_value["effective_config"], baseline_variant),
            ("candidate", candidate_value["effective_config"], candidate_variant),
        ):
            expected = variants[variant_id]["effective_config"]
            if not _json_equal(observed, expected):
                rejected.append(
                    {
                        "compatibility": "effective_config",
                        "side": label,
                        "expected": expected,
                        "observed": observed,
                    }
                )
        baseline_backend = baseline_value["qualification"]["backend"]
        candidate_backend = candidate_value["qualification"]["backend"]
        backend_environment_before = {
            "backend": baseline_backend,
            "environment": baseline_value["environment"],
        }
        backend_environment_after = {
            "backend": candidate_backend,
            "environment": candidate_value["environment"],
        }

        if mode == "same_environment_performance":
            if baseline_variant != candidate_variant:
                rejected.append(
                    {
                        "compatibility": "workload_variant",
                        "baseline": baseline_variant,
                        "candidate": candidate_variant,
                    }
                )
            _add_difference(
                rejected,
                "backend_environment",
                backend_environment_before,
                backend_environment_after,
            )
        elif mode == "declared_ab_variants":
            matched_pair = next(
                (
                    pair
                    for pair in policy["variant_pairs"]
                    if pair["baseline_variant_id"] == baseline_variant
                    and pair["candidate_variant_id"] == candidate_variant
                ),
                None,
            )
            if matched_pair is None:
                compatibility_errors.append(
                    "receipt variants do not match a declared directional A/B variant pair"
                )
                _add_difference(
                    rejected,
                    "workload_variant",
                    baseline_variant,
                    candidate_variant,
                )
            else:
                allowed.append(
                    {
                        "compatibility": "workload_variant",
                        "baseline": baseline_variant,
                        "candidate": candidate_variant,
                    }
                )
            _add_difference(
                rejected,
                "backend_environment",
                backend_environment_before,
                backend_environment_after,
            )
            if matched_pair is not None:
                baseline_config = variants[baseline_variant]["effective_config"]
                candidate_config = variants[candidate_variant]["effective_config"]
                for path in matched_pair["allowed_effective_config_differences"]:
                    before_exists, before = _dot_value(baseline_config, path)
                    after_exists, after = _dot_value(candidate_config, path)
                    assert before_exists and after_exists
                    allowed.append(
                        {
                            "compatibility": "effective_config",
                            "path": path,
                            "baseline": before,
                            "candidate": after,
                        }
                    )
        elif mode == "cross_backend_correctness":
            matched_backend_pair = _matched_backend_pair(
                policy,
                baseline_backend,
                baseline_variant,
                candidate_backend,
                candidate_variant,
            )
            if matched_backend_pair is None:
                compatibility_errors.append(
                    "receipt backend/variant endpoints do not match a declared cross-backend pair"
                )
                _add_difference(
                    rejected,
                    "workload_variant",
                    baseline_variant,
                    candidate_variant,
                )
                _add_difference(
                    rejected,
                    "backend_environment",
                    backend_environment_before,
                    backend_environment_after,
                )
            else:
                allowed.extend(
                    [
                        {
                            "compatibility": "workload_variant",
                            "baseline": baseline_variant,
                            "candidate": candidate_variant,
                        },
                        {
                            "compatibility": "backend",
                            "baseline": baseline_backend,
                            "candidate": candidate_backend,
                        },
                    ]
                )
                environment_allowed, environment_rejected, environment_errors = (
                    _allowed_environment_differences(
                        baseline_value["environment"],
                        candidate_value["environment"],
                        matched_backend_pair["allowed_environment_differences"],
                    )
                )
                allowed.extend(environment_allowed)
                rejected.extend(environment_rejected)
                compatibility_errors.extend(environment_errors)
        else:
            raise ComparisonError(f"unsupported workload comparison mode: {mode!r}")

        evidence_errors.extend(
            _required_result_errors(
                baseline,
                variants[baseline_variant],
                "baseline",
                manifest.value["determinism"]["repetitions"],
            )
        )
        evidence_errors.extend(
            _required_result_errors(
                candidate,
                variants[candidate_variant],
                "candidate",
                manifest.value["determinism"]["repetitions"],
            )
        )
        if baseline_value["metrics"]:
            evidence_errors.append(
                "baseline manifest-bound receipt must keep metrics inside declared results"
            )
        if candidate_value["metrics"]:
            evidence_errors.append(
                "candidate manifest-bound receipt must keep metrics inside declared results"
            )

    if manifest is None:
        if baseline_value["qualification"]["verdict"] != "passed":
            evidence_errors.append("baseline receipt verdict must be passed")
        if candidate_value["qualification"]["verdict"] != "passed":
            evidence_errors.append("candidate receipt verdict must be passed")

    baseline_metrics = _metrics_by_scope(baseline_value)
    candidate_metrics = _metrics_by_scope(candidate_value)
    if policy is not None:
        compatibility_errors.extend(
            _metric_policy_errors(policy, baseline_metrics, candidate_metrics)
        )
    compatible = not rejected and not compatibility_errors
    metric_evaluations: list[dict[str, Any]] = []
    metric_deltas: list[dict[str, Any]] = []
    if compatible and not evidence_errors and policy is not None:
        metric_evaluations = [
            _evaluate_metric_rule(rule, baseline_metrics, candidate_metrics)
            for rule in policy["metric_rules"]
        ]
        if mode != "cross_backend_correctness":
            metric_deltas = copy.deepcopy(metric_evaluations)

    rule_failures = [
        item
        for item in metric_evaluations
        if item["required"] and item["status"] != "passed"
    ]
    result_status_changes: list[dict[str, str]] = []
    if compatible:
        baseline_results = {result["id"]: result for result in baseline_value["results"]}
        candidate_results = {result["id"]: result for result in candidate_value["results"]}
        for result_id in sorted(baseline_results.keys() & candidate_results.keys()):
            before = baseline_results[result_id]["status"]
            after = candidate_results[result_id]["status"]
            if before != after:
                result_status_changes.append(
                    {"id": result_id, "baseline": before, "candidate": after}
                )

    errors = [f"incompatible {item['compatibility']} identity" for item in rejected]
    errors.extend(compatibility_errors)
    errors.extend(evidence_errors)
    errors.extend(
        f"required metric rule failed: {item['scope']}"
        for item in rule_failures
    )
    ok = compatible and not evidence_errors and not rule_failures
    return {
        "ok": ok,
        "compatible": compatible,
        "mode": mode,
        "baseline": {
            "path": str(baseline.path),
            "receipt_id": baseline_value["receipt_id"],
            "sha256": baseline.sha256,
            "verdict": baseline_value["qualification"]["verdict"],
        },
        "candidate": {
            "path": str(candidate.path),
            "receipt_id": candidate_value["receipt_id"],
            "sha256": candidate.sha256,
            "verdict": candidate_value["qualification"]["verdict"],
        },
        "workload_manifest": None
        if manifest is None
        else {
            "path": str(manifest.path),
            "workload_id": manifest.value["workload_id"],
            "sha256": manifest.sha256,
        },
        "compatibility": {
            "rejected_differences": rejected,
            "allowed_differences": allowed,
        },
        "evidence_errors": evidence_errors,
        "errors": errors,
        "metric_evaluations": metric_evaluations,
        "metric_deltas": metric_deltas,
        "result_status_changes": result_status_changes,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        help="exact committed manifest named by both receipts",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _print_text(report: dict[str, Any]) -> None:
    status = "PASSED" if report["ok"] else "FAILED"
    print(
        f"{status} {report['baseline']['receipt_id']} -> "
        f"{report['candidate']['receipt_id']} ({report['mode']})"
    )
    for item in report["compatibility"]["allowed_differences"]:
        suffix = f" {item['path']}" if "path" in item else ""
        print(f"allowed {item['compatibility']}{suffix}")
    for evaluation in report["metric_evaluations"]:
        print(
            f"{evaluation['status'].upper()} {evaluation['scope']}: "
            f"{evaluation['baseline']} -> {evaluation['candidate']}"
        )
    for error in report["errors"]:
        print(f"error: {error}", file=sys.stderr)
    if not report["compatible"]:
        print("metric deltas were not computed", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()

    def rooted(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    try:
        baseline = load_validated_receipt(rooted(args.baseline))
        candidate = load_validated_receipt(rooted(args.candidate))
        manifest = None
        if args.workload_manifest is not None:
            manifest = load_committed_workload(rooted(args.workload_manifest), root=root)
        report = compare_receipts(baseline, candidate, manifest=manifest)
    except ComparisonError as exc:
        if args.json_output:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"comparison failed: {exc}", file=sys.stderr)
        return 1

    report["baseline"]["path"] = _display_path(baseline.path, root)
    report["candidate"]["path"] = _display_path(candidate.path, root)
    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        _print_text(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
