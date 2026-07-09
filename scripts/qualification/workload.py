#!/usr/bin/env python3
"""Validate and fingerprint deterministic local qualification workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
VARIABLE_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
METRIC_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
ENVIRONMENT_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
PLACEHOLDER_RE = re.compile(r"^\$\{([a-z][a-z0-9_]{1,63}|seed)\}$")
RESERVED_PLACEHOLDERS = {"seed", "model_path"}
MODEL_REQUIRED_KINDS = {"serving", "performance", "training", "eval", "soak"}
JSON_INTEGER_MAX_DIGITS = 4096

KINDS = {"environment", "correctness", "serving", "performance", "training", "eval", "soak"}
BACKENDS = {"cpu", "cuda", "rocm", "vulkan", "metal"}
SEED_DELIVERIES = {"not_applicable", "fixed_fixture", "argv", "environment"}
VARIABLE_TYPES = {"string", "integer", "number", "boolean"}
DEVICE_REQUIREMENTS = {"none", "required"}
SKIP_POLICIES = {"allow", "fail"}
OUTPUT_STREAMS = {"stdout", "stderr", "combined"}
OUTPUT_MATCHES = {"required", "forbidden"}
RESULT_PROTOCOL_FORMAT = "qualification-case-result-v1"
RESULT_PROTOCOL_PRODUCERS = {"runner", "command"}
RESULT_PATH_ENVIRONMENT_VARIABLE = "KILN_QUALIFICATION_CASE_RESULT"
RUNNER_METRIC_DEFINITIONS = {
    "case_pass": {"unit": "bool", "aggregation": "exact", "lower_is_better": False},
    "exit_code": {"unit": "code", "aggregation": "exact", "lower_is_better": False},
    "output_assertion_failures": {
        "unit": "count",
        "aggregation": "sum",
        "lower_is_better": True,
    },
}
RUNNER_RESULT_METRICS = set(RUNNER_METRIC_DEFINITIONS)
COMPARISON_MODES = {
    "same_environment_performance",
    "declared_ab_variants",
    "cross_backend_correctness",
}
METRIC_SCOPES = {"result"}
METRIC_CLASSES = {"correctness", "performance"}
METRIC_OPERATORS = {"equal", "not_greater", "not_less"}
CONFIG_PATH_RE = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*$")
CONFIG_SEGMENT_RE = re.compile(r"^[a-z][a-z0-9_-]*$")

TOP_LEVEL_KEYS = {
    "schema_version",
    "workload_id",
    "kind",
    "description",
    "determinism",
    "variables",
    "variants",
    "comparison_policy",
}
DETERMINISM_KEYS = {
    "seed",
    "seed_delivery",
    "repetitions",
    "case_order",
    "max_parallel_cases",
    "network_access",
}
VARIABLE_KEYS = {"name", "description", "type", "required", "default", "constraints"}
CONSTRAINT_KEYS = {"allowed_values", "minimum", "maximum", "pattern"}
VARIANT_KEYS = {
    "id",
    "description",
    "backend",
    "device_requirement",
    "skip_policy",
    "effective_config",
    "cases",
}
CASE_KEYS = {
    "id",
    "description",
    "required",
    "command",
    "working_directory",
    "environment",
    "timeout_seconds",
    "expected_exit_codes",
    "output_assertions",
    "result_protocol",
}
OUTPUT_ASSERTION_KEYS = {"stream", "match", "pattern"}
RESULT_PROTOCOL_KEYS = {"format", "producer", "path_environment_variable", "declared_metrics"}
COMPARISON_KEYS = {
    "mode",
    "variant_pairs",
    "backend_pairs",
    "metric_rules",
}
VARIANT_PAIR_KEYS = {
    "baseline_variant_id",
    "candidate_variant_id",
    "allowed_effective_config_differences",
}
BACKEND_PAIR_KEYS = {
    "backend_a",
    "variant_a_id",
    "backend_b",
    "variant_b_id",
    "allowed_environment_differences",
}
METRIC_RULE_KEYS = {
    "scope",
    "result_id",
    "metric",
    "metric_class",
    "unit",
    "aggregation",
    "lower_is_better",
    "operator",
    "absolute_tolerance",
    "relative_tolerance",
    "required",
}

SHELL_EXECUTABLES = {"sh", "bash", "dash", "zsh", "fish", "pwsh", "powershell", "cmd", "cmd.exe"}
SHELL_EVAL_FLAGS = {"-c", "/c", "-command", "-encodedcommand"}


class WorkloadLoadError(RuntimeError):
    pass


class WorkloadValidationError(RuntimeError):
    pass


def runner_metric_definition(name: str, repetitions: int) -> dict[str, Any] | None:
    del repetitions
    definition = RUNNER_METRIC_DEFINITIONS.get(name)
    if definition is None:
        return None
    return dict(definition)


def _reject_constant(value: str) -> None:
    raise WorkloadLoadError(f"non-finite JSON number is not allowed: {value}")


def _parse_finite_float(value: str) -> float:
    try:
        exact = Decimal(value)
        parsed = float(value)
    except (InvalidOperation, OverflowError, ValueError) as exc:
        raise WorkloadLoadError(f"invalid JSON number: {value}") from exc
    if not math.isfinite(parsed):
        raise WorkloadLoadError(f"JSON number overflows finite float range: {value}")
    if parsed == 0.0:
        if exact != 0:
            raise WorkloadLoadError(f"JSON number underflows finite float range: {value}")
        return 0.0
    if Decimal(str(parsed)) != exact:
        raise WorkloadLoadError(f"JSON number is not exactly representable: {value}")
    return parsed


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > JSON_INTEGER_MAX_DIGITS:
        raise WorkloadLoadError(
            f"JSON integer exceeds {JSON_INTEGER_MAX_DIGITS} digits"
        )
    try:
        return int(value)
    except ValueError as exc:
        raise WorkloadLoadError(f"invalid JSON integer: {value}") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorkloadLoadError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_workload_document(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
            parse_int=_parse_bounded_int,
        )
    except (OSError, json.JSONDecodeError, WorkloadLoadError) as exc:
        raise WorkloadLoadError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkloadLoadError(f"{path}: workload must be a JSON object")
    return value, raw


def load_workload(path: Path) -> dict[str, Any]:
    workload, _ = load_workload_document(path)
    return workload


def _is_number(value: Any) -> bool:
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def _check_exact_keys(errors: list[str], value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{context} must be an object")
        return {}
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing:
        errors.append(f"{context} missing keys: {', '.join(missing)}")
    if unknown:
        errors.append(f"{context} has unknown keys: {', '.join(unknown)}")
    return value


def _check_string(errors: list[str], value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        errors.append(f"{context} must be a non-empty string")
        return ""
    return value


def _check_bool(errors: list[str], value: Any, context: str) -> bool | None:
    if not isinstance(value, bool):
        errors.append(f"{context} must be a boolean")
        return None
    return value


def _check_enum(errors: list[str], value: Any, choices: set[str], context: str) -> str:
    if not isinstance(value, str) or value not in choices:
        errors.append(f"{context} must be one of {sorted(choices)}")
        return ""
    return value


def _check_positive_int(errors: list[str], value: Any, context: str) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{context} must be a positive integer")
        return None
    return value


def _check_nonnegative_number(errors: list[str], value: Any, context: str) -> float | None:
    if not _is_number(value) or value < 0:
        errors.append(f"{context} must be a finite non-negative number")
        return None
    return float(value)


def _scalar_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, json.dumps(value, sort_keys=True, separators=(",", ":"))


def _variable_scalar_key(value: Any, variable_type: str) -> tuple[str, str]:
    if variable_type == "number" and _is_number(value):
        decimal = Decimal(str(value)).normalize()
        if decimal == 0:
            decimal = Decimal(0)
        return "number", str(decimal)
    return _scalar_key(value)


def _matches_variable_type(value: Any, variable_type: str) -> bool:
    if variable_type == "string":
        return isinstance(value, str)
    if variable_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if variable_type == "number":
        return _is_number(value)
    if variable_type == "boolean":
        return isinstance(value, bool)
    return False


def _validate_scalar_constraints(
    errors: list[str],
    value: Any,
    variable_type: str,
    constraints: dict[str, Any],
    context: str,
) -> None:
    if not _matches_variable_type(value, variable_type):
        errors.append(f"{context} must have declared type {variable_type!r}")
        return
    allowed = constraints.get("allowed_values")
    if isinstance(allowed, list) and allowed and not any(
        _variable_scalar_key(value, variable_type) == _variable_scalar_key(item, variable_type)
        for item in allowed
    ):
        errors.append(f"{context} is not one of the declared allowed_values")
    minimum = constraints.get("minimum")
    maximum = constraints.get("maximum")
    if variable_type in {"integer", "number"}:
        if _is_number(minimum) and value < minimum:
            errors.append(f"{context} is less than the declared minimum")
        if _is_number(maximum) and value > maximum:
            errors.append(f"{context} is greater than the declared maximum")
    pattern = constraints.get("pattern")
    if variable_type == "string" and isinstance(pattern, str):
        try:
            matched = re.fullmatch(pattern, value)
        except re.error:
            return
        if matched is None:
            errors.append(f"{context} does not match the declared pattern")


def _validate_config_object(errors: list[str], value: Any, context: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{context} must be an object")
        return
    for key, item in value.items():
        if not isinstance(key, str) or not CONFIG_SEGMENT_RE.fullmatch(key):
            errors.append(f"{context} key {key!r} must be a dot-path-compatible segment")
            continue
        item_context = f"{context}.{key}"
        if isinstance(item, dict):
            _validate_config_object(errors, item, item_context)
        elif item is None or isinstance(item, (str, bool)) or _is_number(item):
            continue
        else:
            errors.append(f"{item_context} must be a finite JSON scalar or nested object")


def _config_difference_paths(baseline: Any, candidate: Any, prefix: str = "") -> list[str]:
    if _scalar_key(baseline) == _scalar_key(candidate):
        return []
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        if baseline.keys() != candidate.keys():
            return [prefix or "<root>"]
        differences: list[str] = []
        for key in sorted(baseline):
            path = f"{prefix}.{key}" if prefix else key
            differences.extend(_config_difference_paths(baseline[key], candidate[key], path))
        return differences
    if isinstance(baseline, dict) or isinstance(candidate, dict):
        return [prefix or "<root>"]
    return [prefix or "<root>"]


def _config_value_at(value: dict[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = value
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return False, None
        current = current[segment]
    return True, current


def _validate_variable(errors: list[str], value: Any, context: str) -> tuple[str, bool | None]:
    variable = _check_exact_keys(errors, value, VARIABLE_KEYS, context)
    name = _check_string(errors, variable.get("name"), f"{context}.name")
    if name and not VARIABLE_RE.fullmatch(name):
        errors.append(f"{context}.name has invalid variable syntax")
    if name in RESERVED_PLACEHOLDERS:
        errors.append(f"{context}.name {name!r} is runner-owned and reserved")
    _check_string(errors, variable.get("description"), f"{context}.description")
    variable_type = _check_enum(
        errors, variable.get("type"), VARIABLE_TYPES, f"{context}.type"
    )
    required = _check_bool(errors, variable.get("required"), f"{context}.required")
    default = variable.get("default")
    if required is True and default is not None:
        errors.append(f"{context}.default must be null when required=true")

    constraints = _check_exact_keys(
        errors, variable.get("constraints"), CONSTRAINT_KEYS, f"{context}.constraints"
    )
    allowed = constraints.get("allowed_values")
    if not isinstance(allowed, list):
        errors.append(f"{context}.constraints.allowed_values must be an array")
        allowed = []
    else:
        keys = [_variable_scalar_key(item, variable_type) for item in allowed]
        if len(keys) != len(set(keys)):
            errors.append(f"{context}.constraints.allowed_values contains duplicates")
        if keys != sorted(keys):
            errors.append(f"{context}.constraints.allowed_values must use canonical sorted order")
        for index, item in enumerate(allowed):
            if variable_type and not _matches_variable_type(item, variable_type):
                errors.append(
                    f"{context}.constraints.allowed_values[{index}] must have declared type {variable_type!r}"
                )

    minimum = constraints.get("minimum")
    maximum = constraints.get("maximum")
    pattern = constraints.get("pattern")
    if variable_type in {"integer", "number"}:
        for field, item in (("minimum", minimum), ("maximum", maximum)):
            if item is not None and not _is_number(item):
                errors.append(f"{context}.constraints.{field} must be null or finite numeric")
        if _is_number(minimum) and _is_number(maximum) and minimum > maximum:
            errors.append(f"{context}.constraints.minimum cannot exceed maximum")
        if pattern is not None:
            errors.append(f"{context}.constraints.pattern is only valid for string variables")
    elif variable_type == "string":
        if minimum is not None or maximum is not None:
            errors.append(f"{context}.constraints.minimum/maximum are only valid for numeric variables")
        if pattern is not None:
            pattern_text = _check_string(errors, pattern, f"{context}.constraints.pattern")
            if pattern_text:
                try:
                    re.compile(pattern_text)
                except re.error as exc:
                    errors.append(f"{context}.constraints.pattern is invalid: {exc}")
    elif variable_type == "boolean":
        if minimum is not None or maximum is not None or pattern is not None:
            errors.append(f"{context}.constraints numeric/pattern fields must be null for boolean variables")

    if default is not None and variable_type:
        _validate_scalar_constraints(errors, default, variable_type, constraints, f"{context}.default")
    return name, required


def _placeholder(
    errors: list[str], value: str, declared_variables: set[str], context: str
) -> str | None:
    match = PLACEHOLDER_RE.fullmatch(value)
    if match:
        name = match.group(1)
        if name not in RESERVED_PLACEHOLDERS and name not in declared_variables:
            errors.append(f"{context} references undeclared variable {name!r}")
        return name
    if "${" in value:
        errors.append(f"{context} placeholders must occupy the entire argv or environment value")
    return None


def _validate_working_directory(errors: list[str], value: Any, context: str) -> None:
    path_text = _check_string(errors, value, context)
    if not path_text:
        return
    if "\\" in path_text:
        errors.append(f"{context} must use repository-relative POSIX syntax")
        return
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{context} must stay within the repository")


def _validate_case(
    errors: list[str],
    value: Any,
    context: str,
    declared_variables: set[str],
) -> tuple[str, bool | None, set[str], set[str], set[str], set[str], str]:
    case = _check_exact_keys(errors, value, CASE_KEYS, context)
    case_id = _check_string(errors, case.get("id"), f"{context}.id")
    if case_id and not ID_RE.fullmatch(case_id):
        errors.append(f"{context}.id has invalid identifier syntax")
    _check_string(errors, case.get("description"), f"{context}.description")
    required = _check_bool(errors, case.get("required"), f"{context}.required")

    command_variables: set[str] = set()
    environment_variables: set[str] = set()
    command = case.get("command")
    if not isinstance(command, list) or not command:
        errors.append(f"{context}.command must be a non-empty argv array")
        command = []
    else:
        for index, argument in enumerate(command):
            argument_text = _check_string(errors, argument, f"{context}.command[{index}]")
            if argument_text:
                name = _placeholder(
                    errors,
                    argument_text,
                    declared_variables,
                    f"{context}.command[{index}]",
                )
                if name:
                    command_variables.add(name)
                if index == 0 and name is not None:
                    errors.append(f"{context}.command[0] executable cannot be a variable")
        executable = command[0] if command and isinstance(command[0], str) else ""
        executable_name = PurePosixPath(executable).name.lower()
        if executable_name in SHELL_EXECUTABLES and any(
            isinstance(argument, str) and argument.lower() in SHELL_EVAL_FLAGS
            for argument in command[1:]
        ):
            errors.append(f"{context}.command cannot use shell command-string evaluation")
    _validate_working_directory(errors, case.get("working_directory"), f"{context}.working_directory")

    environment = case.get("environment")
    if not isinstance(environment, dict):
        errors.append(f"{context}.environment must be an object")
    else:
        for key, item in environment.items():
            if not isinstance(key, str) or not ENVIRONMENT_RE.fullmatch(key):
                errors.append(f"{context}.environment key {key!r} has invalid syntax")
            item_text = _check_string(errors, item, f"{context}.environment.{key}")
            if item_text:
                name = _placeholder(
                    errors, item_text, declared_variables, f"{context}.environment.{key}"
                )
                if name:
                    environment_variables.add(name)
            if key == RESULT_PATH_ENVIRONMENT_VARIABLE:
                errors.append(
                    f"{context}.environment cannot override runner-owned "
                    f"{RESULT_PATH_ENVIRONMENT_VARIABLE}"
                )

    _check_positive_int(errors, case.get("timeout_seconds"), f"{context}.timeout_seconds")
    exit_codes = case.get("expected_exit_codes")
    if not isinstance(exit_codes, list) or not exit_codes:
        errors.append(f"{context}.expected_exit_codes must be a non-empty array")
    else:
        valid_codes: list[int] = []
        for index, code in enumerate(exit_codes):
            if not isinstance(code, int) or isinstance(code, bool) or not 0 <= code <= 255:
                errors.append(f"{context}.expected_exit_codes[{index}] must be an integer from 0 through 255")
            else:
                valid_codes.append(code)
        if len(valid_codes) != len(set(valid_codes)):
            errors.append(f"{context}.expected_exit_codes contains duplicates")
        if valid_codes != sorted(valid_codes):
            errors.append(f"{context}.expected_exit_codes must use ascending order")

    assertions = case.get("output_assertions")
    if not isinstance(assertions, list):
        errors.append(f"{context}.output_assertions must be an array")
    else:
        assertion_keys: set[tuple[Any, Any, Any]] = set()
        for index, raw_assertion in enumerate(assertions):
            assertion_context = f"{context}.output_assertions[{index}]"
            assertion = _check_exact_keys(
                errors, raw_assertion, OUTPUT_ASSERTION_KEYS, assertion_context
            )
            stream = _check_enum(
                errors, assertion.get("stream"), OUTPUT_STREAMS, f"{assertion_context}.stream"
            )
            match_kind = _check_enum(
                errors, assertion.get("match"), OUTPUT_MATCHES, f"{assertion_context}.match"
            )
            pattern = _check_string(errors, assertion.get("pattern"), f"{assertion_context}.pattern")
            if pattern:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    errors.append(f"{assertion_context}.pattern is invalid: {exc}")
            key = (stream, match_kind, pattern)
            if key in assertion_keys:
                errors.append(f"{context}.output_assertions contains a duplicate assertion")
            assertion_keys.add(key)

    protocol = _check_exact_keys(
        errors, case.get("result_protocol"), RESULT_PROTOCOL_KEYS, f"{context}.result_protocol"
    )
    if protocol.get("format") != RESULT_PROTOCOL_FORMAT:
        errors.append(f"{context}.result_protocol.format must be {RESULT_PROTOCOL_FORMAT!r}")
    producer = _check_enum(
        errors,
        protocol.get("producer"),
        RESULT_PROTOCOL_PRODUCERS,
        f"{context}.result_protocol.producer",
    )
    if protocol.get("path_environment_variable") != RESULT_PATH_ENVIRONMENT_VARIABLE:
        errors.append(
            f"{context}.result_protocol.path_environment_variable must be "
            f"{RESULT_PATH_ENVIRONMENT_VARIABLE!r}"
        )
    raw_declared_metrics = protocol.get("declared_metrics")
    declared_metrics: set[str] = set()
    if not isinstance(raw_declared_metrics, list):
        errors.append(f"{context}.result_protocol.declared_metrics must be an array")
    else:
        previous_metric = ""
        for index, metric in enumerate(raw_declared_metrics):
            metric_text = _check_string(
                errors, metric, f"{context}.result_protocol.declared_metrics[{index}]"
            )
            if metric_text and not METRIC_RE.fullmatch(metric_text):
                errors.append(
                    f"{context}.result_protocol.declared_metrics[{index}] has invalid metric syntax"
                )
            if metric_text in declared_metrics:
                errors.append(f"{context}.result_protocol.declared_metrics contains duplicates")
            if metric_text and previous_metric and metric_text < previous_metric:
                errors.append(
                    f"{context}.result_protocol.declared_metrics must use ascending order"
                )
            if metric_text:
                previous_metric = metric_text
                declared_metrics.add(metric_text)
        if producer == "runner":
            unknown_runner_metrics = sorted(declared_metrics - RUNNER_RESULT_METRICS)
            if unknown_runner_metrics:
                errors.append(
                    f"{context}.result_protocol runner cannot produce metrics: "
                    f"{', '.join(unknown_runner_metrics)}"
                )
            if "case_pass" not in declared_metrics:
                errors.append(
                    f"{context}.result_protocol runner must declare the case_pass metric"
                )
        elif producer == "command":
            reserved = sorted(declared_metrics & RUNNER_RESULT_METRICS)
            if reserved:
                errors.append(
                    f"{context}.result_protocol command cannot declare runner-owned metrics: "
                    f"{', '.join(reserved)}"
                )
    return (
        case_id,
        required,
        command_variables | environment_variables,
        command_variables,
        environment_variables,
        declared_metrics,
        producer,
    )


def _validate_comparison(
    errors: list[str],
    value: Any,
    context: str,
    *,
    kind: Any,
    repetitions: Any,
    variants: dict[str, dict[str, Any]],
) -> None:
    if value is None:
        if kind != "environment":
            errors.append(f"{context} may be null only for environment workloads")
        return
    if kind == "environment":
        errors.append(f"{context} must be null for environment workloads")
    comparison = _check_exact_keys(errors, value, COMPARISON_KEYS, context)
    mode = _check_enum(errors, comparison.get("mode"), COMPARISON_MODES, f"{context}.mode")

    raw_variant_pairs = comparison.get("variant_pairs")
    variant_pairs: list[tuple[str, str, tuple[str, ...]]] = []
    if not isinstance(raw_variant_pairs, list):
        errors.append(f"{context}.variant_pairs must be an array")
    else:
        seen_variant_pairs: set[tuple[str, str]] = set()
        for index, raw_pair in enumerate(raw_variant_pairs):
            pair_context = f"{context}.variant_pairs[{index}]"
            pair = _check_exact_keys(errors, raw_pair, VARIANT_PAIR_KEYS, pair_context)
            baseline = _check_string(
                errors, pair.get("baseline_variant_id"), f"{pair_context}.baseline_variant_id"
            )
            candidate = _check_string(
                errors, pair.get("candidate_variant_id"), f"{pair_context}.candidate_variant_id"
            )
            raw_config_paths = pair.get("allowed_effective_config_differences")
            config_paths: list[str] = []
            if not isinstance(raw_config_paths, list) or not raw_config_paths:
                errors.append(
                    f"{pair_context}.allowed_effective_config_differences must be a non-empty array"
                )
            else:
                for path_index, path in enumerate(raw_config_paths):
                    path_context = (
                        f"{pair_context}.allowed_effective_config_differences[{path_index}]"
                    )
                    path_text = _check_string(errors, path, path_context)
                    if path_text and not CONFIG_PATH_RE.fullmatch(path_text):
                        errors.append(f"{path_context} has invalid dot-path syntax")
                    config_paths.append(path_text)
                if len(config_paths) != len(set(config_paths)):
                    errors.append(
                        f"{pair_context}.allowed_effective_config_differences contains duplicates"
                    )
                if config_paths != sorted(config_paths):
                    errors.append(
                        f"{pair_context}.allowed_effective_config_differences must use ascending order"
                    )
            for field, variant_id in (("baseline_variant_id", baseline), ("candidate_variant_id", candidate)):
                if variant_id and not ID_RE.fullmatch(variant_id):
                    errors.append(f"{pair_context}.{field} has invalid identifier syntax")
                elif variant_id and variant_id not in variants:
                    errors.append(f"{pair_context}.{field} names undeclared variant {variant_id!r}")
            if baseline and candidate and baseline == candidate:
                errors.append(f"{pair_context} must name two different variants")
            pair_key = (baseline, candidate)
            if pair_key in seen_variant_pairs:
                errors.append(f"{context}.variant_pairs contains duplicate pair {pair_key!r}")
            seen_variant_pairs.add(pair_key)
            variant_pairs.append((baseline, candidate, tuple(config_paths)))
            if baseline in variants and candidate in variants:
                if variants[baseline]["backend"] != variants[candidate]["backend"]:
                    errors.append(f"{pair_context} A/B variants must use the same backend")
                if variants[baseline]["variables"] != variants[candidate]["variables"]:
                    errors.append(
                        f"{pair_context} A/B variants must reference the same resolved variables"
                    )
                if variants[baseline]["cases"] != variants[candidate]["cases"]:
                    errors.append(
                        f"{pair_context} A/B variants must declare identical result contracts"
                    )
                baseline_config = variants[baseline]["effective_config"]
                candidate_config = variants[candidate]["effective_config"]
                actual_paths = _config_difference_paths(baseline_config, candidate_config)
                if config_paths != actual_paths:
                    errors.append(
                        f"{pair_context}.allowed_effective_config_differences must exactly equal "
                        f"the variant config leaf differences {actual_paths!r}"
                    )
                for path in actual_paths:
                    before_exists, before = _config_value_at(baseline_config, path)
                    after_exists, after = _config_value_at(candidate_config, path)
                    if not before_exists or not after_exists:
                        errors.append(
                            f"{pair_context} config difference {path!r} must exist in both variants"
                        )
                    elif isinstance(before, dict) or isinstance(after, dict):
                        errors.append(
                            f"{pair_context} config difference {path!r} must name a scalar leaf"
                        )
                    elif type(before) is not type(after):
                        errors.append(
                            f"{pair_context} config difference {path!r} must preserve JSON leaf type"
                        )

    raw_backend_pairs = comparison.get("backend_pairs")
    backend_pairs: list[tuple[tuple[str, str], tuple[str, str], tuple[str, ...]]] = []
    if not isinstance(raw_backend_pairs, list):
        errors.append(f"{context}.backend_pairs must be an array")
    else:
        seen_backend_pairs: set[tuple[tuple[str, str], tuple[str, str]]] = set()
        previous_pair: tuple[tuple[str, str], tuple[str, str]] | None = None
        for index, raw_pair in enumerate(raw_backend_pairs):
            pair_context = f"{context}.backend_pairs[{index}]"
            pair = _check_exact_keys(errors, raw_pair, BACKEND_PAIR_KEYS, pair_context)
            backend_a = _check_enum(
                errors, pair.get("backend_a"), BACKENDS, f"{pair_context}.backend_a"
            )
            backend_b = _check_enum(
                errors, pair.get("backend_b"), BACKENDS, f"{pair_context}.backend_b"
            )
            variant_a = _check_string(errors, pair.get("variant_a_id"), f"{pair_context}.variant_a_id")
            variant_b = _check_string(errors, pair.get("variant_b_id"), f"{pair_context}.variant_b_id")
            raw_environment_paths = pair.get("allowed_environment_differences")
            environment_paths: list[str] = []
            if not isinstance(raw_environment_paths, list) or not raw_environment_paths:
                errors.append(
                    f"{pair_context}.allowed_environment_differences must be a non-empty array"
                )
            else:
                for path_index, path in enumerate(raw_environment_paths):
                    path_context = f"{pair_context}.allowed_environment_differences[{path_index}]"
                    path_text = _check_string(errors, path, path_context)
                    if path_text and not CONFIG_PATH_RE.fullmatch(path_text):
                        errors.append(f"{path_context} has invalid dot-path syntax")
                    environment_paths.append(path_text)
                if len(environment_paths) != len(set(environment_paths)):
                    errors.append(f"{pair_context}.allowed_environment_differences contains duplicates")
                if environment_paths != sorted(environment_paths):
                    errors.append(
                        f"{pair_context}.allowed_environment_differences must use ascending order"
                    )
            for endpoint, backend, variant_id in (
                ("a", backend_a, variant_a),
                ("b", backend_b, variant_b),
            ):
                if variant_id and not ID_RE.fullmatch(variant_id):
                    errors.append(f"{pair_context}.variant_{endpoint}_id has invalid identifier syntax")
                elif variant_id not in variants:
                    errors.append(
                        f"{pair_context}.variant_{endpoint}_id names undeclared variant {variant_id!r}"
                    )
                elif backend and variants[variant_id]["backend"] != backend:
                    errors.append(
                        f"{pair_context} endpoint {endpoint} backend does not match variant {variant_id!r}"
                    )
            endpoint_a = (backend_a, variant_a)
            endpoint_b = (backend_b, variant_b)
            pair_key = (endpoint_a, endpoint_b)
            if endpoint_a == endpoint_b:
                errors.append(f"{pair_context} must name two different backend/variant endpoints")
            if backend_a and backend_b and backend_a == backend_b:
                errors.append(f"{pair_context} must name two different backends")
            if endpoint_a > endpoint_b:
                errors.append(f"{pair_context} endpoints must use canonical ascending order")
            if pair_key in seen_backend_pairs:
                errors.append(f"{context}.backend_pairs contains duplicate pair {pair_key!r}")
            if previous_pair is not None and pair_key < previous_pair:
                errors.append(f"{context}.backend_pairs must use canonical ascending order")
            previous_pair = pair_key
            seen_backend_pairs.add(pair_key)
            backend_pairs.append((endpoint_a, endpoint_b, tuple(environment_paths)))
            if variant_a in variants and variant_b in variants:
                if variants[variant_a]["variables"] != variants[variant_b]["variables"]:
                    errors.append(
                        f"{pair_context} cross-backend variants must reference the same resolved variables"
                    )
                if variants[variant_a]["cases"] != variants[variant_b]["cases"]:
                    errors.append(
                        f"{pair_context} cross-backend variants must declare identical result contracts"
                    )
                if not _scalar_key(variants[variant_a]["effective_config"]) == _scalar_key(
                    variants[variant_b]["effective_config"]
                ):
                    errors.append(
                        f"{pair_context} cross-backend variants must declare identical effective_config"
                    )

    rules = comparison.get("metric_rules")
    validated_rules: list[dict[str, Any]] = []
    if not isinstance(rules, list):
        errors.append(f"{context}.metric_rules must be an array")
        rules = []
    identities: set[tuple[Any, Any, Any]] = set()
    for index, raw_rule in enumerate(rules):
        rule_context = f"{context}.metric_rules[{index}]"
        rule = _check_exact_keys(errors, raw_rule, METRIC_RULE_KEYS, rule_context)
        scope = _check_enum(
            errors, rule.get("scope"), METRIC_SCOPES, f"{rule_context}.scope"
        )
        result_id = rule.get("result_id")
        if scope == "result":
            if not isinstance(result_id, str) or not ID_RE.fullmatch(result_id):
                errors.append(f"{rule_context}.result_id must be a valid identifier for result scope")
        metric = _check_string(errors, rule.get("metric"), f"{rule_context}.metric")
        if metric and not METRIC_RE.fullmatch(metric):
            errors.append(f"{rule_context}.metric has invalid metric syntax")
        metric_class = _check_enum(
            errors,
            rule.get("metric_class"),
            METRIC_CLASSES,
            f"{rule_context}.metric_class",
        )
        unit = _check_string(errors, rule.get("unit"), f"{rule_context}.unit")
        aggregation = _check_string(
            errors, rule.get("aggregation"), f"{rule_context}.aggregation"
        )
        lower_is_better = _check_bool(
            errors, rule.get("lower_is_better"), f"{rule_context}.lower_is_better"
        )
        operator = _check_enum(
            errors, rule.get("operator"), METRIC_OPERATORS, f"{rule_context}.operator"
        )
        if operator == "not_greater" and lower_is_better is not True:
            errors.append(f"{rule_context}.operator='not_greater' requires lower_is_better=true")
        if operator == "not_less" and lower_is_better is not False:
            errors.append(f"{rule_context}.operator='not_less' requires lower_is_better=false")
        expected_runner_definition = runner_metric_definition(
            metric, repetitions if isinstance(repetitions, int) else 1
        )
        if expected_runner_definition is not None:
            observed_definition = {
                "unit": unit,
                "aggregation": aggregation,
                "lower_is_better": lower_is_better,
            }
            if observed_definition != expected_runner_definition:
                errors.append(
                    f"{rule_context} runner metric {metric!r} must use canonical definition "
                    f"{expected_runner_definition!r}"
                )
        _check_nonnegative_number(
            errors, rule.get("absolute_tolerance"), f"{rule_context}.absolute_tolerance"
        )
        _check_nonnegative_number(
            errors, rule.get("relative_tolerance"), f"{rule_context}.relative_tolerance"
        )
        _check_bool(errors, rule.get("required"), f"{rule_context}.required")
        identity = (scope, result_id if isinstance(result_id, str) else "", metric)
        if identity in identities:
            errors.append(f"{context}.metric_rules contains duplicate metric identity {identity!r}")
        identities.add(identity)
        validated_rules.append(rule)

    if validated_rules and not any(rule.get("required") is True for rule in validated_rules):
        errors.append(f"{context}.metric_rules must contain at least one required rule")

    if mode == "same_environment_performance":
        if kind != "performance":
            errors.append(f"{context}.mode same_environment_performance requires kind='performance'")
        if variant_pairs or backend_pairs:
            errors.append(
                f"{context}.mode same_environment_performance cannot declare variant or backend pairs"
            )
        if not validated_rules:
            errors.append(f"{context}.mode same_environment_performance requires metric_rules")
    elif mode == "declared_ab_variants":
        if kind != "performance":
            errors.append(f"{context}.mode declared_ab_variants requires kind='performance'")
        if not variant_pairs:
            errors.append(f"{context}.mode declared_ab_variants requires variant_pairs")
        if backend_pairs:
            errors.append(f"{context}.mode declared_ab_variants cannot declare backend_pairs")
        if not validated_rules:
            errors.append(f"{context}.mode declared_ab_variants requires metric_rules")
    elif mode == "cross_backend_correctness":
        if kind != "correctness":
            errors.append(f"{context}.mode cross_backend_correctness requires kind='correctness'")
        if variant_pairs:
            errors.append(f"{context}.mode cross_backend_correctness cannot declare variant_pairs")
        if not backend_pairs:
            errors.append(f"{context}.mode cross_backend_correctness requires backend_pairs")
        if not validated_rules:
            errors.append(f"{context}.mode cross_backend_correctness requires metric_rules")

    comparison_variant_ids: set[str]
    if mode == "declared_ab_variants":
        comparison_variant_ids = {
            variant_id
            for baseline, candidate, _config_paths in variant_pairs
            for variant_id in (baseline, candidate)
        }
    elif mode == "cross_backend_correctness":
        comparison_variant_ids = {
            endpoint[1]
            for endpoint_a, endpoint_b, _environment_paths in backend_pairs
            for endpoint in (endpoint_a, endpoint_b)
        }
    else:
        comparison_variant_ids = set(variants)

    if mode in {"same_environment_performance", "declared_ab_variants"}:
        has_command_performance_evidence = any(
            rule.get("required") is True
            and rule.get("metric_class") == "performance"
            and rule.get("metric") not in RUNNER_RESULT_METRICS
            and isinstance(rule.get("result_id"), str)
            and all(
                variants.get(variant_id, {}).get("cases", {})
                .get(rule["result_id"], {})
                .get("producer")
                == "command"
                for variant_id in comparison_variant_ids
            )
            for rule in validated_rules
        )
        if not has_command_performance_evidence:
            errors.append(
                f"{context} performance comparison requires at least one required "
                "command-produced non-runner metric rule"
            )

    for index, rule in enumerate(validated_rules):
        rule_context = f"{context}.metric_rules[{index}]"
        if mode == "cross_backend_correctness":
            if rule.get("scope") != "result":
                errors.append(f"{rule_context} cross-backend correctness metrics must be result-scoped")
            if rule.get("metric_class") != "correctness":
                errors.append(f"{rule_context} cross-backend comparison forbids performance metrics")
            if rule.get("required") is not True:
                errors.append(f"{rule_context} cross-backend correctness metrics must be required")
            if rule.get("operator") != "equal":
                errors.append(f"{rule_context} cross-backend correctness metrics must use operator='equal'")
        elif rule.get("metric_class") != "performance":
            errors.append(f"{rule_context} performance comparison requires metric_class='performance'")

        if rule.get("scope") == "result" and isinstance(rule.get("result_id"), str):
            result_id = rule["result_id"]
            metric = rule.get("metric")
            for variant_id in sorted(comparison_variant_ids):
                variant = variants.get(variant_id)
                if variant is None:
                    continue
                case = variant["cases"].get(result_id)
                if case is None:
                    errors.append(
                        f"{rule_context}.result_id {result_id!r} is not declared by variant {variant_id!r}"
                    )
                elif case["required"] is not True:
                    errors.append(
                        f"{rule_context}.result_id {result_id!r} is not required by variant {variant_id!r}"
                    )
                elif metric not in case["metrics"]:
                    errors.append(
                        f"{rule_context}.metric {metric!r} is not declared by result {result_id!r} "
                        f"in variant {variant_id!r}"
                    )


def validate_workload(workload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    top = _check_exact_keys(errors, workload, TOP_LEVEL_KEYS, "workload")
    if top.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"workload.schema_version must be {SCHEMA_VERSION}")
    workload_id = _check_string(errors, top.get("workload_id"), "workload.workload_id")
    if workload_id and not ID_RE.fullmatch(workload_id):
        errors.append("workload.workload_id has invalid identifier syntax")
    kind = _check_enum(errors, top.get("kind"), KINDS, "workload.kind")
    _check_string(errors, top.get("description"), "workload.description")

    determinism = _check_exact_keys(
        errors, top.get("determinism"), DETERMINISM_KEYS, "workload.determinism"
    )
    seed = determinism.get("seed")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
    ):
        errors.append("workload.determinism.seed must be null or a non-negative integer")
    seed_delivery = _check_enum(
        errors,
        determinism.get("seed_delivery"),
        SEED_DELIVERIES,
        "workload.determinism.seed_delivery",
    )
    _check_positive_int(errors, determinism.get("repetitions"), "workload.determinism.repetitions")
    if determinism.get("case_order") != "declared":
        errors.append("workload.determinism.case_order must be 'declared'")
    if determinism.get("max_parallel_cases") != 1:
        errors.append("workload.determinism.max_parallel_cases must be 1")
    if determinism.get("network_access") != "forbidden":
        errors.append("workload.determinism.network_access must be 'forbidden'")
    if kind == "environment":
        if seed is not None or seed_delivery != "not_applicable":
            errors.append(
                "environment workloads require seed=null and seed_delivery='not_applicable'"
            )
    else:
        if seed is None:
            errors.append(f"{kind!r} workloads require an explicit deterministic seed")
        if seed_delivery == "not_applicable":
            errors.append(f"{kind!r} workloads cannot use seed_delivery='not_applicable'")
        if seed_delivery == "fixed_fixture" and kind != "correctness":
            errors.append("seed_delivery='fixed_fixture' is limited to correctness workloads")

    variables = top.get("variables")
    variable_names: set[str] = set()
    required_variables: set[str] = set()
    if not isinstance(variables, list):
        errors.append("workload.variables must be an array")
    else:
        previous_name = ""
        for index, raw_variable in enumerate(variables):
            name, required = _validate_variable(errors, raw_variable, f"workload.variables[{index}]")
            if name in variable_names:
                errors.append(f"workload.variables contains duplicate name {name!r}")
            if name and previous_name and name < previous_name:
                errors.append("workload.variables must use ascending name order")
            if name:
                previous_name = name
                variable_names.add(name)
                if required:
                    required_variables.add(name)

    variants = top.get("variants")
    variant_ids: set[str] = set()
    validated_variants: dict[str, dict[str, Any]] = {}
    all_references: set[str] = set()
    variant_command_seed: dict[str, bool] = {}
    variant_environment_seed: dict[str, bool] = {}
    variant_model_path: dict[str, bool] = {}
    if not isinstance(variants, list) or not variants:
        errors.append("workload.variants must be a non-empty array")
    else:
        previous_variant_id = ""
        for variant_index, raw_variant in enumerate(variants):
            variant_context = f"workload.variants[{variant_index}]"
            variant = _check_exact_keys(errors, raw_variant, VARIANT_KEYS, variant_context)
            variant_id = _check_string(errors, variant.get("id"), f"{variant_context}.id")
            if variant_id and not ID_RE.fullmatch(variant_id):
                errors.append(f"{variant_context}.id has invalid identifier syntax")
            if variant_id in variant_ids:
                errors.append(f"workload.variants contains duplicate id {variant_id!r}")
            if variant_id and previous_variant_id and variant_id < previous_variant_id:
                errors.append("workload.variants must use ascending id order")
            if variant_id:
                previous_variant_id = variant_id
                variant_ids.add(variant_id)
            _check_string(errors, variant.get("description"), f"{variant_context}.description")
            backend = _check_enum(
                errors, variant.get("backend"), BACKENDS, f"{variant_context}.backend"
            )
            device_requirement = _check_enum(
                errors,
                variant.get("device_requirement"),
                DEVICE_REQUIREMENTS,
                f"{variant_context}.device_requirement",
            )
            skip_policy = _check_enum(
                errors,
                variant.get("skip_policy"),
                SKIP_POLICIES,
                f"{variant_context}.skip_policy",
            )
            if backend in BACKENDS - {"cpu"} and device_requirement != "required":
                errors.append(f"{variant_context} accelerator backend requires device_requirement='required'")
            if device_requirement == "required" and skip_policy != "fail":
                errors.append(f"{variant_context} required device cannot allow skipped cases")

            effective_config = variant.get("effective_config")
            _validate_config_object(
                errors, effective_config, f"{variant_context}.effective_config"
            )

            cases = variant.get("cases")
            case_ids: set[str] = set()
            references: set[str] = set()
            case_contracts: dict[str, dict[str, Any]] = {}
            command_seed = False
            environment_seed = False
            required_case_count = 0
            if not isinstance(cases, list) or not cases:
                errors.append(f"{variant_context}.cases must be a non-empty array")
            else:
                for case_index, raw_case in enumerate(cases):
                    (
                        case_id,
                        required,
                        case_references,
                        command_references,
                        environment_references,
                        declared_metrics,
                        producer,
                    ) = _validate_case(
                        errors,
                        raw_case,
                        f"{variant_context}.cases[{case_index}]",
                        variable_names,
                    )
                    if case_id in case_ids:
                        errors.append(f"{variant_context}.cases contains duplicate id {case_id!r}")
                    case_ids.add(case_id)
                    if case_id:
                        case_contracts[case_id] = {
                            "required": required,
                            "metrics": declared_metrics,
                            "producer": producer,
                        }
                    required_case_count += required is True
                    references.update(case_references)
                    command_seed = command_seed or "seed" in command_references
                    environment_seed = environment_seed or "seed" in environment_references
            if required_case_count == 0:
                errors.append(f"{variant_context}.cases must contain at least one required case")
            if effective_config and any(
                case_contract.get("producer") != "command"
                for case_contract in case_contracts.values()
            ):
                errors.append(
                    f"{variant_context} non-empty effective_config requires command-produced cases"
                )
            parameter_references = references - RESERVED_PLACEHOLDERS
            missing_required = sorted(required_variables - parameter_references)
            if missing_required:
                errors.append(
                    f"{variant_context} does not consume required variables: {', '.join(missing_required)}"
                )
            all_references.update(parameter_references)
            variant_command_seed[variant_id] = command_seed
            variant_environment_seed[variant_id] = environment_seed
            variant_model_path[variant_id] = "model_path" in references
            if variant_id:
                validated_variants[variant_id] = {
                    "backend": backend,
                    "cases": case_contracts,
                    "effective_config": effective_config,
                    "variables": parameter_references,
                }

    unused_variables = sorted(variable_names - all_references)
    if unused_variables:
        errors.append(f"workload.variables contains unused definitions: {', '.join(unused_variables)}")
    if seed_delivery == "argv":
        for variant_id, used in variant_command_seed.items():
            if not used:
                errors.append(f"workload variant {variant_id!r} does not deliver determinism.seed through argv")
    elif seed_delivery == "environment":
        for variant_id, used in variant_environment_seed.items():
            if not used:
                errors.append(
                    f"workload variant {variant_id!r} does not deliver determinism.seed through environment"
                )
    elif seed_delivery in {"fixed_fixture", "not_applicable"}:
        if any(variant_command_seed.values()) or any(variant_environment_seed.values()):
            errors.append(f"seed_delivery={seed_delivery!r} cannot reference the reserved ${{seed}} placeholder")

    if kind == "environment":
        for variant_id, used in variant_model_path.items():
            if used:
                errors.append(
                    f"workload variant {variant_id!r} cannot reference ${{model_path}} "
                    "for an environment workload"
                )
    elif kind in MODEL_REQUIRED_KINDS or any(variant_model_path.values()):
        for variant_id, used in variant_model_path.items():
            if not used:
                errors.append(
                    f"workload variant {variant_id!r} must consume the runner-owned "
                    "${model_path} placeholder"
                )

    _validate_comparison(
        errors,
        top.get("comparison_policy"),
        "workload.comparison_policy",
        kind=kind,
        repetitions=determinism.get("repetitions"),
        variants=validated_variants,
    )
    return errors


def validate_workload_parameters(
    workload: dict[str, Any], parameters: Any
) -> list[str]:
    """Validate the exact resolved parameter set for one selected variant."""
    errors: list[str] = []
    if not isinstance(parameters, dict):
        return ["workload.parameters must be an object"]
    variant_id = parameters.get("variant_id")
    variants = {
        item.get("id"): item
        for item in workload.get("variants", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if not isinstance(variant_id, str) or variant_id not in variants:
        return ["workload.parameters.variant_id must name a declared variant"]

    references: set[str] = set()
    variant = variants[variant_id]
    for case in variant.get("cases", []):
        if not isinstance(case, dict):
            continue
        values = list(case.get("command", []))
        environment = case.get("environment", {})
        if isinstance(environment, dict):
            values.extend(environment.values())
        for value in values:
            if isinstance(value, str):
                match = PLACEHOLDER_RE.fullmatch(value)
                if match and match.group(1) not in RESERVED_PLACEHOLDERS:
                    references.add(match.group(1))

    expected_keys = {"variant_id"} | references
    actual_keys = set(parameters)
    missing = sorted(expected_keys - actual_keys)
    unknown = sorted(actual_keys - expected_keys)
    if missing:
        errors.append(f"workload.parameters missing resolved keys: {', '.join(missing)}")
    if unknown:
        errors.append(f"workload.parameters has undeclared keys: {', '.join(unknown)}")

    variables = {
        item.get("name"): item
        for item in workload.get("variables", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    for name in sorted(references & actual_keys):
        variable = variables.get(name)
        if variable is None:
            errors.append(f"workload.parameters.{name} has no variable declaration")
            continue
        constraints = variable.get("constraints")
        if not isinstance(constraints, dict):
            constraints = {}
        if variable.get("type") == "number" and not isinstance(parameters[name], float):
            errors.append(
                f"workload.parameters.{name} must use canonical JSON float representation"
            )
            continue
        _validate_scalar_constraints(
            errors,
            parameters[name],
            variable.get("type", ""),
            constraints,
            f"workload.parameters.{name}",
        )
    return errors


def workload_file_sha256(path: Path) -> str:
    workload, raw = load_workload_document(path)
    errors = validate_workload(workload)
    if errors:
        raise WorkloadValidationError("invalid workload:\n  - " + "\n  - ".join(errors))
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workloads", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    report: list[dict[str, Any]] = []
    failed = False
    for workload_path in args.workloads:
        path = workload_path if workload_path.is_absolute() else root / workload_path
        try:
            workload, raw = load_workload_document(path)
        except WorkloadLoadError as exc:
            errors = [str(exc)]
            workload_id = None
            digest = None
            variants: list[str] = []
        else:
            errors = validate_workload(workload)
            workload_id = workload.get("workload_id")
            digest = f"sha256:{hashlib.sha256(raw).hexdigest()}" if not errors else None
            raw_variants = workload.get("variants")
            variants = [
                item["id"]
                for item in raw_variants
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            ] if isinstance(raw_variants, list) else []
        failed = failed or bool(errors)
        report.append(
            {
                "path": str(workload_path),
                "ok": not errors,
                "workload_id": workload_id,
                "sha256": digest,
                "variants": variants,
                "errors": errors,
            }
        )

    if args.json_output:
        print(json.dumps({"ok": not failed, "workloads": report}, indent=2, sort_keys=True))
    else:
        for item in report:
            if item["ok"]:
                print(f"OK {item['path']} {item['sha256']}")
            else:
                print(f"FAILED {item['path']}", file=sys.stderr)
                for error in item["errors"]:
                    print(f"  - {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
