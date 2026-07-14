"""Dependency-free validation for Kiln's checked-in JSON Schema contracts.

This intentionally implements only the Draft 2020-12 keywords used by Kiln's
contract fixtures. Unsupported keywords remain annotations; callers own exact
contract-metadata checks for any product-specific extensions.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


class SchemaResolutionError(ValueError):
    pass


def _pointer(document: Any, fragment: str, reference: str) -> dict[str, Any]:
    current = document
    if fragment:
        if not fragment.startswith("/"):
            raise SchemaResolutionError(f"unsupported schema fragment in {reference!r}")
        for part in fragment[1:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(current, dict) or key not in current:
                raise SchemaResolutionError(f"unresolved schema reference {reference!r}")
            current = current[key]
    if not isinstance(current, dict):
        raise SchemaResolutionError(f"schema reference {reference!r} is not an object")
    return current


def resolve_ref(
    schema: dict[str, Any],
    root: dict[str, Any],
    registry: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved, _ = _resolve_ref_context(schema, root, registry)
    return resolved


def _resolve_ref_context(
    schema: dict[str, Any],
    root: dict[str, Any],
    registry: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = schema
    document = root
    seen: set[tuple[int, str]] = set()
    while "$ref" in current:
        reference = current["$ref"]
        if not isinstance(reference, str):
            raise SchemaResolutionError(f"unsupported schema reference {reference!r}")
        identity = (id(document), reference)
        if identity in seen:
            raise SchemaResolutionError(f"cyclic schema reference {reference!r}")
        seen.add(identity)
        document_name, separator, fragment = reference.partition("#")
        if not separator and document_name:
            fragment = ""
        if document_name:
            if registry is None or document_name not in registry:
                raise SchemaResolutionError(f"unregistered schema reference {reference!r}")
            document = registry[document_name]
        current = _pointer(document, fragment, reference)
    return current, document


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if expected == "null":
        return value is None
    return False


def validate_instance(
    value: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str = "$",
    *,
    registry: Mapping[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Return validation errors for the supported JSON Schema subset."""

    try:
        resolved, active_root = _resolve_ref_context(schema, root, registry)
    except SchemaResolutionError as error:
        return [f"{path}: {error}"]
    errors: list[str] = []

    for subschema in resolved.get("allOf", []):
        errors.extend(validate_instance(value, subschema, active_root, path, registry=registry))

    if "anyOf" in resolved:
        if not any(
            not validate_instance(value, option, active_root, path, registry=registry)
            for option in resolved["anyOf"]
        ):
            errors.append(f"{path}: must match at least one anyOf branch")

    if "oneOf" in resolved:
        matches = sum(
            not validate_instance(value, option, active_root, path, registry=registry)
            for option in resolved["oneOf"]
        )
        if matches != 1:
            errors.append(f"{path}: must match exactly one oneOf branch")

    if "not" in resolved and not validate_instance(
        value, resolved["not"], active_root, path, registry=registry
    ):
        errors.append(f"{path}: must not match the forbidden schema")

    condition = resolved.get("if")
    if isinstance(condition, dict):
        condition_matches = not validate_instance(
            value, condition, active_root, path, registry=registry
        )
        branch = resolved.get("then" if condition_matches else "else")
        if isinstance(branch, dict):
            errors.extend(validate_instance(value, branch, active_root, path, registry=registry))

    if "const" in resolved and value != resolved["const"]:
        errors.append(f"{path}: must equal {resolved['const']!r}")
    if "enum" in resolved and value not in resolved["enum"]:
        errors.append(f"{path}: must be one of {resolved['enum']!r}")

    expected_type = resolved.get("type")
    if isinstance(expected_type, str) and not _type_matches(value, expected_type):
        errors.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
        return errors
    if isinstance(expected_type, list) and not any(
        isinstance(item, str) and _type_matches(value, item) for item in expected_type
    ):
        errors.append(f"{path}: expected one of {expected_type!r}, got {type(value).__name__}")
        return errors

    if isinstance(value, dict):
        required = resolved.get("required", [])
        for name in required:
            if name not in value:
                errors.append(f"{path}: missing required property {name}")
        properties = resolved.get("properties", {})
        additional = resolved.get("additionalProperties", True)
        property_names = resolved.get("propertyNames")
        for name, item in value.items():
            item_path = f"{path}.{name}"
            if name in properties:
                errors.extend(
                    validate_instance(
                        item, properties[name], active_root, item_path, registry=registry
                    )
                )
            elif additional is False:
                errors.append(f"{item_path}: unknown property")
            elif isinstance(additional, dict):
                errors.extend(
                    validate_instance(item, additional, active_root, item_path, registry=registry)
                )
            if isinstance(property_names, dict):
                errors.extend(
                    validate_instance(
                        name, property_names, active_root, item_path, registry=registry
                    )
                )
        if "minProperties" in resolved and len(value) < resolved["minProperties"]:
            errors.append(f"{path}: has fewer than {resolved['minProperties']} properties")
        if "maxProperties" in resolved and len(value) > resolved["maxProperties"]:
            errors.append(f"{path}: has more than {resolved['maxProperties']} properties")
        for names in resolved.get("dependentRequired", {}).items():
            name, dependencies = names
            if name in value:
                for dependency in dependencies:
                    if dependency not in value:
                        errors.append(f"{path}: {name} requires property {dependency}")
        for ordering in resolved.get("x-kiln-dependent-order", []):
            low = ordering["less-or-equal"]
            high = ordering["greater-or-equal"]
            if low in value and high in value and value[low] > value[high]:
                errors.append(f"{path}.{high}: must be at least {path}.{low}")
        toggle = resolved.get("x-kiln-compatible-mode-toggle")
        if isinstance(toggle, dict) and toggle["mode"] in value and toggle["toggle"] in value:
            expected = toggle["true-value"] if value[toggle["toggle"]] else toggle["false-value"]
            if value[toggle["mode"]] != expected:
                errors.append(
                    f"{path}.{toggle['mode']}: conflicts with deprecated {path}.{toggle['toggle']}"
                )

    if isinstance(value, list):
        if "minItems" in resolved and len(value) < resolved["minItems"]:
            errors.append(f"{path}: has fewer than {resolved['minItems']} items")
        if "maxItems" in resolved and len(value) > resolved["maxItems"]:
            errors.append(f"{path}: has more than {resolved['maxItems']} items")
        if resolved.get("uniqueItems") is True:
            encoded = [repr(item) for item in value]
            if len(encoded) != len(set(encoded)):
                errors.append(f"{path}: items must be unique")
        items = resolved.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                errors.extend(
                    validate_instance(
                        item, items, active_root, f"{path}[{index}]", registry=registry
                    )
                )

    if isinstance(value, str):
        if "minLength" in resolved and len(value) < resolved["minLength"]:
            errors.append(f"{path}: shorter than minLength {resolved['minLength']}")
        if "maxLength" in resolved and len(value) > resolved["maxLength"]:
            errors.append(f"{path}: longer than maxLength {resolved['maxLength']}")
        if "pattern" in resolved and re.search(resolved["pattern"], value) is None:
            errors.append(f"{path}: does not match pattern {resolved['pattern']}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in resolved and value < resolved["minimum"]:
            errors.append(f"{path}: below minimum {resolved['minimum']}")
        if "maximum" in resolved and value > resolved["maximum"]:
            errors.append(f"{path}: above maximum {resolved['maximum']}")
        if "exclusiveMinimum" in resolved and value <= resolved["exclusiveMinimum"]:
            errors.append(f"{path}: must exceed {resolved['exclusiveMinimum']}")
        if "exclusiveMaximum" in resolved and value >= resolved["exclusiveMaximum"]:
            errors.append(f"{path}: must be below {resolved['exclusiveMaximum']}")
        if "multipleOf" in resolved and value % resolved["multipleOf"] != 0:
            errors.append(f"{path}: must be a multiple of {resolved['multipleOf']}")
    return errors
