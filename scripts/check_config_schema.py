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
    "paths",
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

PROFILE_GATES = {
    "accelerator.kt_api_mode": {
        "profile": "experimental",
        "when": {"enum": ["all", "disabled"]},
    },
    "accelerator.vulkan_validation": {
        "profile": "experimental",
        "when": {"const": True},
    },
    "accelerator.cuda_marlin_profile": {
        "profile": "experimental",
        "when": {"enum": ["attention_mlp", "attention_mlp_gdn"]},
    },
    "accelerator.rocm_synchronization_mode": {
        "profile": "experimental",
        "when": {"const": "stream_ordered"},
    },
    "accelerator.rocm_strided_batched_matmul_mode": {
        "profile": "experimental",
        "when": {"enum": ["enabled", "disabled"]},
    },
    "accelerator.rocm_bf16_matmul_output_mode": {
        "profile": "experimental",
        "when": {"enum": ["native_bf16", "f32_then_cast"]},
    },
    "accelerator.rocm_kernel_profile": {
        "profile": "experimental",
        "when": {"const": "experimental_multiblock"},
    },
    "accelerator.rocm_graph_mode": {
        "profile": "experimental",
        "when": {"enum": ["warmup_then_eager", "lazy_capture_replay"]},
    },
    "memory.kv_force_blocks": {
        "profile": "maintenance",
        "when": {"minimum": 1},
    },
}

OPERATIONAL_SOURCE_ROOTS = (".github", "scripts", "capabilities", "desktop")
OPERATIONAL_SOURCE_SUFFIXES = {
    ".js",
    ".mjs",
    ".py",
    ".rs",
    ".sh",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
OPERATIONAL_SOURCE_IGNORED_DIRECTORIES = {
    "__pycache__",
    "archive",
    "node_modules",
    "target",
}
RETIRED_ENV_REFERENCE_ALLOWLIST = {
    "scripts/check_docs_site_smoke.mjs": "asserts the published retirement index",
    "scripts/check_server_ui_smoke.mjs": "asserts retired names remain inert",
    "scripts/h15c_kiln_alpha_from_csv.py": "records an exact historical benchmark invocation",
}


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


def retired_environment_replacements() -> dict[str, str]:
    replacements: dict[str, str] = {}
    pattern = re.compile(r"^\| `(KILN_[A-Z0-9_]+)` \| `(KILN_[A-Z0-9_]+)` \|$")
    for line in REFERENCE_PATH.read_text().splitlines():
        match = pattern.match(line)
        if not match:
            continue
        retired, canonical = match.groups()
        if retired in replacements:
            raise ContractError(f"duplicate retired environment name {retired}")
        replacements[retired] = canonical
    return replacements


def find_retired_environment_names(
    text: str, retired: dict[str, str]
) -> list[str]:
    if not retired:
        return []
    alternatives = "|".join(
        re.escape(name) for name in sorted(retired, key=len, reverse=True)
    )
    pattern = re.compile(rf"(?<![A-Z0-9_])(?:{alternatives})(?![A-Z0-9_])")
    return sorted({match.group(0) for match in pattern.finditer(text)})


def validate_operational_retired_environment_references(
    retired: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    seen_allowlist: set[str] = set()
    for root_name in OPERATIONAL_SOURCE_ROOTS:
        source_root = ROOT / root_name
        if not source_root.is_dir():
            errors.append(f"operational source root is missing: {root_name}")
            continue
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or path.suffix not in OPERATIONAL_SOURCE_SUFFIXES:
                continue
            relative = path.relative_to(ROOT)
            if any(
                part in OPERATIONAL_SOURCE_IGNORED_DIRECTORIES
                for part in relative.parts
            ):
                continue
            relative_name = relative.as_posix()
            try:
                found = find_retired_environment_names(path.read_text(), retired)
            except (OSError, UnicodeDecodeError) as error:
                errors.append(f"cannot scan operational source {relative_name}: {error}")
                continue
            if not found:
                continue
            if relative_name in RETIRED_ENV_REFERENCE_ALLOWLIST:
                seen_allowlist.add(relative_name)
                continue
            for name in found:
                errors.append(
                    f"{relative_name} references retired environment name {name}; "
                    f"use {retired[name]}"
                )

    missing_allowlist = sorted(set(RETIRED_ENV_REFERENCE_ALLOWLIST) - seen_allowlist)
    for relative_name in missing_allowlist:
        errors.append(
            f"retired environment reference allowlist entry is stale: {relative_name}"
        )
    return errors


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


def validate_profile_gates(schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    fields = schema_fields(schema)
    declared = {
        path: field["x-kiln-profile-gate"]
        for path, field in fields.items()
        if "x-kiln-profile-gate" in field
    }
    if declared != PROFILE_GATES:
        missing = sorted(set(PROFILE_GATES) - set(declared))
        extra = sorted(set(declared) - set(PROFILE_GATES))
        changed = sorted(
            path
            for path in set(PROFILE_GATES) & set(declared)
            if PROFILE_GATES[path] != declared[path]
        )
        errors.append(
            "profile-gated field metadata drifted"
            f" (missing={missing}, extra={extra}, changed={changed})"
        )

    conditional_gates: dict[str, dict[str, Any]] = {}
    all_of = schema.get("allOf")
    if not isinstance(all_of, list):
        return errors + ["schema allOf must enumerate every profile gate"]
    for index, rule in enumerate(all_of):
        if not isinstance(rule, dict):
            errors.append(f"schema allOf[{index}] must be an object")
            continue
        if_clause = rule.get("if")
        then_clause = rule.get("then")
        if not isinstance(if_clause, dict) or not isinstance(then_clause, dict):
            errors.append(f"schema allOf[{index}] must contain object if/then clauses")
            continue
        root_properties = if_clause.get("properties")
        if not isinstance(root_properties, dict) or len(root_properties) != 1:
            errors.append(f"schema allOf[{index}] must gate exactly one section")
            continue
        section, section_clause = next(iter(root_properties.items()))
        if not isinstance(section_clause, dict):
            errors.append(f"schema allOf[{index}] section clause must be an object")
            continue
        field_properties = section_clause.get("properties")
        if not isinstance(field_properties, dict) or len(field_properties) != 1:
            errors.append(f"schema allOf[{index}] must gate exactly one field")
            continue
        field, condition = next(iter(field_properties.items()))
        path = f"{section}.{field}"
        expected_if = {
            "properties": {
                section: {
                    "properties": {field: condition},
                    "required": [field],
                }
            },
            "required": [section],
        }
        if if_clause != expected_if:
            errors.append(f"schema profile condition for {path} is not fail-closed")

        server_clause = then_clause.get("properties", {}).get("server")
        if not isinstance(server_clause, dict):
            errors.append(f"schema profile condition for {path} has no server gate")
            continue
        profile = server_clause.get("properties", {}).get("serving_profile", {}).get("const")
        expected_server = {
            "properties": {"serving_profile": {"const": profile}},
            "required": ["serving_profile"],
        }
        if server_clause != expected_server or "server" not in then_clause.get("required", []):
            errors.append(f"schema profile condition for {path} does not require the profile")
            continue
        if path in conditional_gates:
            errors.append(f"duplicate schema profile condition for {path}")
            continue
        conditional_gates[path] = {"profile": profile, "when": condition}

    if conditional_gates != PROFILE_GATES:
        errors.append("schema allOf profile conditions drifted from field metadata")
    return errors


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
    alternate_environment = {
        path: cells[3] for path, cells in fixed_rows.items() if cells[3].lower() != "none"
    }
    if alternate_environment:
        errors.append(
            "configuration fields must not document alternate environment spellings: "
            + ", ".join(sorted(alternate_environment))
        )
    counts = {
        "x-kiln-field-count": len(fixed_rows),
        "x-kiln-dynamic-field-template-count": len(dynamic_rows),
        "x-kiln-canonical-environment-count": len(implemented),
        "x-kiln-config-file-only-count": len(fixed_rows) - len(implemented),
        "x-kiln-compatibility-field-count": 0,
        "x-kiln-compatibility-alias-count": 0,
        "x-kiln-toml-compatibility-field-count": 0,
    }
    for key, expected in counts.items():
        if schema.get(key) != expected:
            errors.append(f"schema {key} must be {expected}, got {schema.get(key)!r}")

    retired = schema.get("x-kiln-retired-environment-replacements")
    if not isinstance(retired, dict) or not all(
        isinstance(name, str) and isinstance(replacement, str)
        for name, replacement in retired.items()
    ):
        errors.append("schema retired environment replacements must be a string map")
        retired = {}
    documented_retired = retired_environment_replacements()
    if documented_retired != retired:
        errors.append("retired environment replacement index drifted from the schema")
    if schema.get("x-kiln-retired-environment-count") != len(retired):
        errors.append(
            "schema retired environment count must match its replacement index"
        )
    canonical_names = {
        re.search(r"KILN_[A-Z0-9_]+", cells[2]).group(0)
        for cells in implemented.values()
    }
    if set(retired) & canonical_names:
        errors.append("retired environment names must be disjoint from canonical names")
    unknown_replacements = sorted(set(retired.values()) - canonical_names)
    if unknown_replacements:
        errors.append(
            "retired environment replacements must name canonical overrides: "
            + ", ".join(unknown_replacements)
        )
    errors.extend(validate_operational_retired_environment_references(retired))

    removed_toml = schema.get("x-kiln-removed-toml-field-replacements")
    expected_removed_toml = {
        "speculative.enabled": "speculative.method",
        "streaming_prefill.enabled": "streaming_prefill.mode",
    }
    if removed_toml != expected_removed_toml:
        errors.append("schema removed TOML field replacement index is incomplete")
    if schema.get("x-kiln-removed-toml-field-count") != len(expected_removed_toml):
        errors.append("schema removed TOML field count must match its replacement index")

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

    for removed_path in expected_removed_toml:
        if removed_path in fields:
            errors.append(f"schema must reject removed TOML field {removed_path}")

    errors.extend(validate_profile_gates(schema))

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
            {"model": {"checkpoint_read_mib_per_second": 256}},
            True,
            "bounded checkpoint read rate",
        ),
        (
            {"model": {"checkpoint_read_mib_per_second": 0}},
            False,
            "zero checkpoint read rate",
        ),
        (
            {"model": {"accelerator_weight_upload_mib_per_second": 256}},
            True,
            "bounded accelerator weight upload rate",
        ),
        (
            {"model": {"accelerator_weight_upload_mib_per_second": 0}},
            False,
            "zero accelerator weight upload rate",
        ),
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
                "server": {"serving_profile": "experimental"},
                "accelerator": {"rocm_kernel_profile": "experimental_multiblock"},
            },
            True,
            "experimental ROCm kernel profile",
        ),
        (
            {"accelerator": {"rocm_kernel_profile": "experimental_multiblock"}},
            False,
            "experimental ROCm kernel default profile rejection",
        ),
        (
            {"accelerator": {"rocm_kernel_profile": "portable_fallback"}},
            True,
            "portable ROCm fallback profile",
        ),
        (
            {"accelerator": {"cuda_kernel_profile": "native_default"}},
            True,
            "native-default CUDA backend profile",
        ),
        (
            {"accelerator": {"cuda_kernel_profile": "portable_fallback"}},
            True,
            "portable CUDA backend fallback profile",
        ),
        (
            {"accelerator": {"cuda_kernel_profile": "individual_switches"}},
            False,
            "invalid CUDA backend profile",
        ),
        (
            {"accelerator": {"cuda_marlin_profile": "disabled"}},
            True,
            "disabled CUDA Marlin layout",
        ),
        (
            {"accelerator": {"cuda_marlin_profile": "attention_mlp_gdn"}},
            False,
            "CUDA Marlin default profile rejection",
        ),
        (
            {
                "server": {"serving_profile": "experimental"},
                "accelerator": {"cuda_marlin_profile": "attention_mlp_gdn"},
            },
            True,
            "expanded CUDA Marlin experimental layout",
        ),
        (
            {"accelerator": {"cuda_marlin_profile": "everything"}},
            False,
            "invalid CUDA Marlin layout",
        ),
        (
            {"accelerator": {"cuda_flash_backward_mode": "fast"}},
            True,
            "fast CUDA FlashAttention backward",
        ),
        (
            {"accelerator": {"cuda_flash_backward_mode": "deterministic"}},
            True,
            "deterministic CUDA FlashAttention backward",
        ),
        (
            {"accelerator": {"cuda_flash_backward_mode": "auto"}},
            False,
            "invalid CUDA FlashAttention backward mode",
        ),
        (
            {"accelerator": {"metal_kernel_profile": "native_default"}},
            True,
            "native-default Metal backend profile",
        ),
        (
            {"accelerator": {"metal_kernel_profile": "portable_fallback"}},
            True,
            "portable Metal backend fallback profile",
        ),
        (
            {"accelerator": {"metal_kernel_profile": "individual_switches"}},
            False,
            "invalid Metal backend profile",
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
        ({"streaming_prefill": {"enabled": False}}, False, "removed streaming toggle"),
        ({"speculative": {"enabled": False}}, False, "removed speculative toggle"),
    ]
    errors = []
    for value, expected_valid, label in cases:
        actual_valid = not validate_instance(value, schema, schema)
        if actual_valid != expected_valid:
            errors.append(f"self-test {label!r} expected valid={expected_valid}, got {actual_valid}")
    synthetic_retired = {"KILN_OLD": "KILN_NEW"}
    found = find_retired_environment_names(
        "KILN_OLDER=1 KILN_OLD=1 NOT_KILN_OLD=1", synthetic_retired
    )
    if found != ["KILN_OLD"]:
        errors.append(f"self-test retired environment token matching got {found!r}")
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
        f"{schema['x-kiln-compatibility-alias-count']} compatibility aliases, "
        f"{len(PROFILE_GATES)} profile gates, "
        "0 executable retired environment references"
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
