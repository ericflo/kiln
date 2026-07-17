#!/usr/bin/env python3
"""Validate the canonical Kiln configuration schema and its published inputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

from json_schema_subset import validate_instance


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "contracts" / "kiln-config-v1.schema.json"
REFERENCE_PATH = ROOT / "docs" / "CONFIGURATION.md"
EXAMPLE_PATH = ROOT / "kiln.example.toml"
SECTIONS = (
    "server",
    "accelerator",
    "batching",
    "model",
    "memory",
    "training",
    "logging",
    "prefix_cache",
    "speculative",
    "streaming_prefill",
    "adapters",
    "teachers",
    "eval",
    "request_log",
    "agent",
)


class ContractError(Exception):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {path.relative_to(ROOT)}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def reference_rows() -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in REFERENCE_PATH.read_text().splitlines():
        match = re.match(r"^\| `([^`]+)` \|", line)
        if not match or match.group(1).split(".", 1)[0] not in SECTIONS:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 5:
            raise ContractError(
                f"{REFERENCE_PATH.relative_to(ROOT)} field row {match.group(1)} must have five columns"
            )
        path = match.group(1)
        if path in rows:
            raise ContractError(f"duplicate configuration reference row {path}")
        rows[path] = cells
    return rows


def schema_fields(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for definition in schema.get("$defs", {}).values():
        if not isinstance(definition, dict):
            continue
        for field in definition.get("properties", {}).values():
            if not isinstance(field, dict):
                continue
            path = field.get("x-kiln-path")
            if not isinstance(path, str):
                continue
            if path in result:
                raise ContractError(f"duplicate x-kiln-path in schema: {path}")
            result[path] = field
    return result


def strip_markup(value: str) -> str:
    return value.replace("`", "")


def compatibility_aliases(cell: str) -> list[str]:
    if "deprecated" not in cell:
        return []
    return re.findall(r"`(KILN_[A-Z0-9_]+)`", cell)


def validate_contract_metadata(schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_identity = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://ericflo.github.io/kiln/contracts/kiln-config-v1.schema.json",
        "type": "object",
        "additionalProperties": False,
    }
    for key, expected in expected_identity.items():
        if schema.get(key) != expected:
            errors.append(f"schema {key} must be {expected!r}, got {schema.get(key)!r}")

    root_properties = schema.get("properties")
    if not isinstance(root_properties, dict):
        errors.append("schema properties must be an object")
        root_properties = {}
    if tuple(root_properties) != SECTIONS:
        errors.append(
            "schema root sections must exactly match the typed order: " + ", ".join(SECTIONS)
        )
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        errors.append("schema $defs must be an object")
        definitions = {}
    for section in SECTIONS:
        expected_ref = {"$ref": f"#/$defs/{section}"}
        if root_properties.get(section) != expected_ref:
            errors.append(f"root property {section} must be {expected_ref}")
        definition = definitions.get(section)
        if not isinstance(definition, dict):
            errors.append(f"missing object definition for {section}")
            continue
        if definition.get("type") != "object" or definition.get("additionalProperties") is not False:
            errors.append(f"definition {section} must be a closed object")

    rows = reference_rows()
    fields = schema_fields(schema)
    documented = set(rows)
    canonical = {path for path, field in fields.items() if not field.get("deprecated")}
    if documented != canonical:
        missing = sorted(documented - canonical)
        extra = sorted(canonical - documented)
        if missing:
            errors.append("documented fields missing from schema: " + ", ".join(missing))
        if extra:
            errors.append("canonical schema fields missing from reference: " + ", ".join(extra))

    fixed_rows = {path: cells for path, cells in rows.items() if "<id>" not in path}
    dynamic_rows = {path: cells for path, cells in rows.items() if "<id>" in path}
    implemented = {
        path: cells for path, cells in fixed_rows.items() if "(implemented)" in cells[2]
    }
    compatibility = {
        path: compatibility_aliases(cells[3]) for path, cells in fixed_rows.items()
    }
    compatibility = {path: aliases for path, aliases in compatibility.items() if aliases}
    counts = {
        "x-kiln-field-count": len(fixed_rows),
        "x-kiln-dynamic-field-template-count": len(dynamic_rows),
        "x-kiln-canonical-environment-count": len(implemented),
        "x-kiln-config-file-only-count": len(fixed_rows) - len(implemented),
        "x-kiln-compatibility-field-count": len(compatibility),
        "x-kiln-compatibility-alias-count": sum(map(len, compatibility.values())),
        "x-kiln-toml-compatibility-field-count": 1,
    }
    for key, expected in counts.items():
        if schema.get(key) != expected:
            errors.append(f"schema {key} must be {expected}, got {schema.get(key)!r}")

    metadata_keys = (
        "x-kiln-type-and-default",
        "x-kiln-canonical-env",
        "x-kiln-environment",
        "x-kiln-validation",
    )
    for path, cells in rows.items():
        field = fields.get(path)
        if field is None:
            continue
        expected_values = tuple(strip_markup(value) for value in cells[1:5])
        for key, expected in zip(metadata_keys, expected_values, strict=True):
            if field.get(key) != expected:
                errors.append(f"{path} {key} drifted from the reference table")
        if path in implemented:
            section, name = path.split(".", 1)
            expected_name = f"KILN_{section.upper()}_{name.upper()}"
            if not field["x-kiln-canonical-env"].startswith(expected_name):
                errors.append(
                    f"{path} canonical environment name must be mechanically derived as {expected_name}"
                )

    legacy = fields.get("streaming_prefill.enabled", {})
    if legacy.get("deprecated") is not True or legacy.get("type") != "boolean":
        errors.append("schema must expose deprecated TOML streaming_prefill.enabled as a boolean")

    credential = definitions.get("teacher_credential", {})
    if credential.get("required") != ["origin", "api_key_env"]:
        errors.append("teacher_credential must require origin and api_key_env")
    credentials = definitions.get("teachers", {}).get("properties", {}).get("credentials", {})
    if credentials.get("additionalProperties") != {"$ref": "#/$defs/teacher_credential"}:
        errors.append("teachers.credentials values must reference teacher_credential")

    return errors


def validate_defaults(schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for path, field in schema_fields(schema).items():
        if "default" not in field:
            continue
        for error in validate_instance(field["default"], field, schema, f"default({path})"):
            errors.append(error)
    return errors


def run_self_tests(schema: dict[str, Any]) -> list[str]:
    cases = [
        ({}, True, "empty config uses typed defaults"),
        ({"unknown": {}}, False, "unknown root section"),
        ({"server": {"porrt": 8420}}, False, "unknown field"),
        ({"server": {"port": True}}, False, "boolean is not an integer"),
        ({"server": {"max_decode_batch": "auto"}}, True, "auto union"),
        ({"server": {"max_decode_batch": 0}}, False, "bounded union"),
        (
            {"teachers": {"credentials": {"judge": {"origin": "https://judge.test"}}}},
            False,
            "credential required fields",
        ),
        (
            {
                "server": {"serving_profile": "experimental"},
                "accelerator": {"rocm_synchronization_mode": "stream_ordered"},
            },
            True,
            "stream-ordered experimental profile",
        ),
        (
            {"accelerator": {"rocm_synchronization_mode": "stream_ordered"}},
            False,
            "stream-ordered default profile rejection",
        ),
        (
            {
                "server": {"serving_profile": "experimental"},
                "accelerator": {"kt_api_mode": "disabled"},
            },
            True,
            "explicit tensor API mode experimental profile",
        ),
        (
            {"accelerator": {"kt_api_mode": "all"}},
            False,
            "explicit tensor API mode default profile rejection",
        ),
        (
            {
                "server": {"serving_profile": "experimental"},
                "accelerator": {"vulkan_validation": True},
            },
            True,
            "Vulkan validation experimental profile",
        ),
        (
            {"accelerator": {"vulkan_validation": True}},
            False,
            "Vulkan validation default profile rejection",
        ),
        (
            {
                "server": {"serving_profile": "maintenance"},
                "memory": {"kv_force_blocks": 8, "kv_autoscale": True},
            },
            True,
            "forced KV maintenance policy",
        ),
        ({"memory": {"kv_force_blocks": 8}}, False, "forced KV policy rejection"),
        (
            {"training": {"max_queued_jobs": 8, "max_tracked_jobs": 4}},
            False,
            "dependent queue ordering",
        ),
        (
            {"streaming_prefill": {"mode": "enabled", "enabled": False}},
            False,
            "legacy TOML conflict",
        ),
    ]
    errors = []
    for value, expected_valid, label in cases:
        actual_valid = not validate_instance(value, schema, schema)
        if actual_valid != expected_valid:
            errors.append(f"self-test {label!r} expected valid={expected_valid}, got {actual_valid}")
    return errors


def check(*, self_test: bool) -> None:
    schema = load_json(SCHEMA_PATH)
    errors = validate_contract_metadata(schema)
    errors.extend(validate_defaults(schema))
    try:
        example = tomllib.loads(EXAMPLE_PATH.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        errors.append(f"cannot read {EXAMPLE_PATH.relative_to(ROOT)}: {error}")
    else:
        errors.extend(validate_instance(example, schema, schema, "kiln.example.toml"))
    if self_test:
        errors.extend(run_self_tests(schema))
    if errors:
        raise ContractError("configuration schema contract failed:\n- " + "\n- ".join(errors))
    print(
        "configuration schema contract passed: "
        f"{schema['x-kiln-field-count']} canonical fields, "
        f"{schema['x-kiln-dynamic-field-template-count']} dynamic templates, "
        f"{schema['x-kiln-canonical-environment-count']} canonical environment overrides, "
        f"{schema['x-kiln-compatibility-alias-count']} compatibility aliases"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="also run validator mutation cases")
    args = parser.parse_args()
    try:
        check(self_test=args.self_test)
    except ContractError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
