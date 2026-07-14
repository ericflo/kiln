#!/usr/bin/env python3
"""Validate Kiln's canonical OpenAPI operation and transport contract."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from json_schema_subset import SchemaResolutionError, resolve_ref, validate_instance


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts" / "kiln-http-api-v1.openapi.json"
INFERENCE_SCHEMA_PATH = ROOT / "contracts" / "kiln-inference-v1.schema.json"
OBSERVABILITY_SCHEMA_PATH = ROOT / "contracts" / "kiln-observability-v1.schema.json"
THINKING_SCHEMA_PATH = ROOT / "contracts" / "thinking-budget-v1.schema.json"
INFERENCE_ENTRYPOINTS = (
    "BatchCompletionRequest",
    "BatchCompletionResponse",
    "ChatCompletionChunkStream",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "TextCompletionRequest",
    "TextCompletionResponse",
)
OBSERVABILITY_ENTRYPOINTS = (
    "CacheStatsResponse",
    "ConfigResponse",
    "DebugDisabledResponse",
    "DebugProvenanceErrorResponse",
    "DecodeStatsSnapshot",
    "HealthResponse",
    "ModelStateResponse",
    "ModelsResponse",
    "RequestRecord",
    "Vec_RequestRecord",
)
OBSERVABILITY_COMPONENT_TYPES = {
    "CacheStatsResponse": "CacheStatsResponse",
    "ConfigResponse": "ConfigResponse",
    "DebugDisabledResponse": "DebugDisabledResponse",
    "DebugProvenanceErrorResponse": "serde_json::Value",
    "DecodeStatsSnapshot": "DecodeStatsSnapshot",
    "HealthResponse": "HealthResponse",
    "ModelStateResponse": "ModelStateResponse",
    "ModelsResponse": "ModelsResponse",
    "RequestRecord": "RequestRecord",
    "Vec_RequestRecord": "Vec<RequestRecord>",
}
EXPECTED_COMPONENT_SCHEMA_COUNTS = {
    "complete": 27,
    "migration_pending": 80,
    "total": 107,
}
HTTP_METHODS = ("get", "post", "put", "patch", "delete")
EXPECTED_METHOD_COUNTS = {"DELETE": 12, "GET": 53, "POST": 47}
EXPECTED_TAG_COUNTS = {
    "adapters": 9,
    "agents": 16,
    "corrections": 5,
    "evals": 25,
    "hf-trl": 7,
    "inference": 3,
    "library": 3,
    "observability": 8,
    "preflight": 5,
    "recipes": 2,
    "teachers": 5,
    "training": 15,
    "ui": 9,
}
ALLOWED_MEDIA_TYPES = {
    "application/gzip",
    "application/javascript",
    "application/json",
    "application/octet-stream",
    "multipart/form-data",
    "text/css",
    "text/event-stream",
    "text/html",
    "text/plain",
}
NO_BODY_POSTS = {
    "/v1/adapters/unload",
    "/v1/agent/runs/{id}/abort",
    "/v1/library/install/{id}",
}
EXPLICIT_ERROR_PATHS = {"/v1/debug/model-state", "/v1/health"}


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


def load_contract() -> dict[str, Any]:
    return load_json(CONTRACT_PATH)


def iter_operations(document: dict[str, Any]):
    paths = document.get("paths", {})
    if not isinstance(paths, dict):
        return
    for path, item in paths.items():
        if not isinstance(item, dict):
            continue
        for method in HTTP_METHODS:
            operation = item.get(method)
            if isinstance(operation, dict):
                yield path, method, operation


def resolve_contract_ref(document: dict[str, Any], reference: str) -> Any:
    document_name, separator, fragment = reference.partition("#")
    if not separator:
        fragment = ""
    if document_name:
        external_documents = {
            INFERENCE_SCHEMA_PATH.name: INFERENCE_SCHEMA_PATH,
            OBSERVABILITY_SCHEMA_PATH.name: OBSERVABILITY_SCHEMA_PATH,
        }
        if document_name not in external_documents:
            raise ContractError(f"unsupported external OpenAPI reference {reference!r}")
        value: Any = load_json(external_documents[document_name])
    else:
        value = document
    if fragment and not fragment.startswith("/"):
        raise ContractError(f"unsupported OpenAPI reference fragment {reference!r}")
    for part in fragment[1:].split("/") if fragment else ():
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or key not in value:
            raise ContractError(f"unresolved OpenAPI reference {reference!r}")
        value = value[key]
    return value


def schema_rust_type(document: dict[str, Any], schema: Any) -> str | None:
    if not isinstance(schema, dict):
        return None
    value = schema.get("x-kiln-rust-type")
    if isinstance(value, str):
        return value
    if "$ref" in schema:
        try:
            resolved = resolve_contract_ref(document, schema["$ref"])
        except ContractError:
            return None
        return resolved.get("x-kiln-rust-type") if isinstance(resolved, dict) else None
    return None


def content_media_types(content: Any) -> list[str]:
    return list(content) if isinstance(content, dict) else []


def validate_contract(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_root = {
        "openapi": "3.1.1",
        "jsonSchemaDialect": "https://json-schema.org/draft/2020-12/schema",
        "x-kiln-path-count": 101,
        "x-kiln-operation-count": 112,
        "x-kiln-method-counts": EXPECTED_METHOD_COUNTS,
        "x-kiln-tag-counts": EXPECTED_TAG_COUNTS,
        "x-kiln-field-schema-status": "migration_pending",
        "x-kiln-component-schema-counts": EXPECTED_COMPONENT_SCHEMA_COUNTS,
    }
    for key, expected in expected_root.items():
        if document.get(key) != expected:
            errors.append(f"root {key} must be {expected!r}, got {document.get(key)!r}")

    info = document.get("info")
    if not isinstance(info, dict):
        errors.append("info must be an object")
    else:
        for key in ("title", "version", "description"):
            if not isinstance(info.get(key), str) or not info[key].strip():
                errors.append(f"info.{key} must be a non-empty string")
    servers = document.get("servers")
    if servers != [{"url": "http://127.0.0.1:8420"}]:
        errors.append("servers must contain only the documented local default origin")

    declared_tags = document.get("tags")
    if not isinstance(declared_tags, list):
        errors.append("tags must be an array")
        declared_tag_names: list[str] = []
    else:
        declared_tag_names = []
        for index, tag in enumerate(declared_tags):
            if not isinstance(tag, dict):
                errors.append(f"tags[{index}] must be an object")
                continue
            name = tag.get("name")
            description = tag.get("description")
            if not isinstance(name, str) or not name:
                errors.append(f"tags[{index}].name must be non-empty")
            else:
                declared_tag_names.append(name)
            if not isinstance(description, str) or not description:
                errors.append(f"tags[{index}].description must be non-empty")
        if len(declared_tag_names) != len(set(declared_tag_names)):
            errors.append("tag names must be unique")
        if set(declared_tag_names) != set(EXPECTED_TAG_COUNTS):
            errors.append("declared tags must exactly match the operation tag contract")

    paths = document.get("paths")
    if not isinstance(paths, dict):
        errors.append("paths must be an object")
        return errors
    if len(paths) != 101:
        errors.append(f"paths must contain 101 entries, got {len(paths)}")
    if list(paths) != sorted(paths):
        errors.append("paths must be sorted lexicographically")

    operation_ids: set[str] = set()
    method_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    websocket_operations = []
    operation_count = 0
    for path, method, operation in iter_operations(document):
        operation_count += 1
        method_counts[method.upper()] += 1
        label = f"{method.upper()} {path}"
        if not path.startswith("/") or "//" in path:
            errors.append(f"{label}: path must be absolute and normalized")
        unknown_item_keys = set(paths[path]) - set(HTTP_METHODS)
        if unknown_item_keys:
            errors.append(f"{path}: unsupported path-item keys {sorted(unknown_item_keys)}")

        operation_id = operation.get("operationId")
        if not isinstance(operation_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", operation_id):
            errors.append(f"{label}: operationId must be lower snake case")
        elif operation_id in operation_ids:
            errors.append(f"{label}: duplicate operationId {operation_id}")
        else:
            operation_ids.add(operation_id)
        if not isinstance(operation.get("summary"), str) or not operation["summary"].strip():
            errors.append(f"{label}: summary must be non-empty")

        tags = operation.get("tags")
        if not isinstance(tags, list) or len(tags) != 1 or tags[0] not in declared_tag_names:
            errors.append(f"{label}: exactly one declared tag is required")
        else:
            tag_counts[tags[0]] += 1

        handler = operation.get("x-kiln-handler")
        signature = operation.get("x-kiln-rust-signature")
        if not isinstance(handler, str) or not re.fullmatch(r"[a-z0-9_]+::[a-z0-9_]+", handler):
            errors.append(f"{label}: x-kiln-handler must be module::handler")
        if not isinstance(signature, str) or not isinstance(handler, str):
            errors.append(f"{label}: x-kiln-rust-signature must be present")
        elif not signature.startswith(f"async fn {handler.split('::')[1]}("):
            errors.append(f"{label}: signature does not match x-kiln-handler")

        transport = operation.get("x-kiln-transport")
        if transport not in {"http", "websocket"}:
            errors.append(f"{label}: x-kiln-transport must be http or websocket")
        if transport == "websocket":
            websocket_operations.append(label)

        placeholders = re.findall(r"\{([^{}]+)\}", path)
        parameters = operation.get("parameters", [])
        if not isinstance(parameters, list):
            errors.append(f"{label}: parameters must be an array")
            parameters = []
        path_parameters = []
        seen_parameters = set()
        for parameter in parameters:
            if not isinstance(parameter, dict):
                errors.append(f"{label}: every parameter must be an object")
                continue
            key = (parameter.get("in"), parameter.get("name"))
            if key in seen_parameters:
                errors.append(f"{label}: duplicate parameter {key}")
            seen_parameters.add(key)
            if parameter.get("in") == "path":
                path_parameters.append(parameter.get("name"))
                if parameter.get("required") is not True:
                    errors.append(f"{label}: path parameters must be required")
                schema = parameter.get("schema", {})
                if schema.get("type") != "string" or schema.get("minLength") != 1:
                    errors.append(f"{label}: path parameters must be non-empty strings")
            elif parameter.get("in") == "header":
                if parameter.get("schema") != {"type": "string"}:
                    errors.append(f"{label}: header parameters must be strings")
            else:
                errors.append(f"{label}: only path and header parameter records are allowed")
        if path_parameters != placeholders:
            errors.append(
                f"{label}: path parameters {path_parameters!r} do not match placeholders {placeholders!r}"
            )

        query_type = operation.get("x-kiln-query-rust-type")
        signature_has_query = isinstance(signature, str) and "Query<" in signature
        if signature_has_query != isinstance(query_type, str):
            errors.append(f"{label}: query Rust type metadata does not match the handler signature")

        request_body = operation.get("requestBody")
        signature_has_body = isinstance(signature, str) and any(
            marker in signature for marker in ("Json<", "Json(mut ", "Multipart", "body: Body")
        )
        expects_body = method == "post" and path not in NO_BODY_POSTS and signature_has_body
        if expects_body != isinstance(request_body, dict):
            errors.append(f"{label}: requestBody presence does not match the handler contract")
        if method != "post" and request_body is not None:
            errors.append(f"{label}: only POST operations may declare requestBody")
        if isinstance(request_body, dict):
            if request_body.get("required") is not True:
                errors.append(f"{label}: declared request bodies must be required")
            rust_type = request_body.get("x-kiln-rust-type")
            content = request_body.get("content")
            media_types = content_media_types(content)
            if len(media_types) != 1 or media_types[0] not in ALLOWED_MEDIA_TYPES:
                errors.append(f"{label}: request body must have one supported media type")
            elif schema_rust_type(document, content[media_types[0]].get("schema")) != rust_type:
                errors.append(f"{label}: request body schema does not match x-kiln-rust-type")

        responses = operation.get("responses")
        if not isinstance(responses, dict):
            errors.append(f"{label}: responses must be an object")
            continue
        success = [code for code in responses if code.isdigit() and 100 <= int(code) < 400]
        if len(success) != 1:
            errors.append(f"{label}: exactly one success response is required")
        else:
            response = responses[success[0]]
            if not isinstance(response, dict):
                errors.append(f"{label}: success response must be an object")
            else:
                rust_type = response.get("x-kiln-rust-type")
                content = response.get("content")
                media_types = content_media_types(content)
                if not media_types or any(value not in ALLOWED_MEDIA_TYPES for value in media_types):
                    errors.append(f"{label}: success response media types are missing or unsupported")
                for media_type in media_types:
                    media = content.get(media_type)
                    if not isinstance(media, dict) or not isinstance(media.get("schema"), dict):
                        errors.append(f"{label}: {media_type} response must declare a schema")
                if len(media_types) == 1:
                    actual = schema_rust_type(document, content[media_types[0]]["schema"])
                    if actual != rust_type:
                        errors.append(f"{label}: response schema does not match x-kiln-rust-type")
                headers = response.get("headers", {})
                if not isinstance(headers, dict) or any(
                    value != {"schema": {"type": "string"}} for value in headers.values()
                ):
                    errors.append(f"{label}: response headers must be string schemas")
        if path.startswith("/v1/") and path not in EXPLICIT_ERROR_PATHS and responses.get("default") != {
            "$ref": "#/components/responses/ApiError"
        }:
            errors.append(f"{label}: /v1 operations must reference the structured default error")
        if path in EXPLICIT_ERROR_PATHS and "default" in responses:
            errors.append(f"{label}: explicit error responses must not retain a fictitious default error")

    if operation_count != 112:
        errors.append(f"operation count must be 112, got {operation_count}")
    if dict(sorted(method_counts.items())) != EXPECTED_METHOD_COUNTS:
        errors.append(f"observed method counts drifted: {dict(sorted(method_counts.items()))}")
    if dict(sorted(tag_counts.items())) != EXPECTED_TAG_COUNTS:
        errors.append(f"observed tag counts drifted: {dict(sorted(tag_counts.items()))}")
    if websocket_operations != ["GET /v1/terminal/ws"]:
        errors.append(f"WebSocket transport must be exactly GET /v1/terminal/ws, got {websocket_operations}")

    explicit_responses = {
        "/health": {
            "200": ("HealthResponse", "HealthResponse"),
            "503": ("HealthResponse", "HealthResponse"),
        },
        "/v1/health": {
            "200": ("HealthResponse", "HealthResponse"),
            "503": ("HealthResponse", "HealthResponse"),
        },
        "/v1/debug/model-state": {
            "200": ("ModelStateResponse", "ModelStateResponse"),
            "403": ("DebugDisabledResponse", "DebugDisabledResponse"),
            "500": ("DebugProvenanceErrorResponse", "serde_json::Value"),
        },
    }
    for path, expected in explicit_responses.items():
        responses = paths.get(path, {}).get("get", {}).get("responses", {})
        if set(responses) != set(expected):
            errors.append(f"GET {path}: response statuses must be exactly {sorted(expected)}")
            continue
        for code, (component, rust_type) in expected.items():
            response = responses[code]
            schema = response.get("content", {}).get("application/json", {}).get("schema", {})
            expected_ref = f"#/components/schemas/{component}"
            if schema.get("$ref") != expected_ref:
                errors.append(f"GET {path} {code}: response schema must be {expected_ref}")
            if response.get("x-kiln-rust-type") != rust_type:
                errors.append(f"GET {path} {code}: response Rust type must be {rust_type}")

    components = document.get("components")
    if not isinstance(components, dict):
        errors.append("components must be an object")
        return errors
    schemas = components.get("schemas")
    if not isinstance(schemas, dict) or not schemas:
        errors.append("components.schemas must be a non-empty object")
        schemas = {}
    if list(schemas) != sorted(schemas):
        errors.append("components.schemas must be sorted")
    api_error = schemas.get("ApiError")
    if not isinstance(api_error, dict) or api_error.get("additionalProperties") is not False:
        errors.append("ApiError must be a closed component schema")
    status_counts: Counter[str] = Counter()
    for name, schema in schemas.items():
        if not isinstance(schema, dict):
            errors.append(f"component schema {name} must be an object")
            continue
        rust_type = schema.get("x-kiln-rust-type")
        if not isinstance(rust_type, str) or not rust_type:
            errors.append(f"component schema {name} must declare x-kiln-rust-type")
        status = schema.get("x-kiln-field-schema-status")
        if status not in {"complete", "migration_pending"}:
            errors.append(f"component schema {name} must declare a supported field-schema status")
            continue
        status_counts[status] += 1
        if status == "migration_pending":
            description = schema.get("description", "")
            if "migration" not in description.lower():
                errors.append(f"migration-pending component schema {name} must state its status")
        elif "Field-level schema migration is tracked separately" in schema.get("description", ""):
            errors.append(f"complete component schema {name} retains a stale migration description")
    observed_counts = {
        "complete": status_counts["complete"],
        "migration_pending": status_counts["migration_pending"],
        "total": len(schemas),
    }
    if observed_counts != EXPECTED_COMPONENT_SCHEMA_COUNTS:
        errors.append(f"component field-schema counts drifted: {observed_counts}")
    for entrypoint in INFERENCE_ENTRYPOINTS:
        schema = schemas.get(entrypoint, {})
        expected_ref = f"{INFERENCE_SCHEMA_PATH.name}#/$defs/{entrypoint}"
        if schema.get("$ref") != expected_ref or schema.get("x-kiln-field-schema-status") != "complete":
            errors.append(f"inference component {entrypoint} must use complete ref {expected_ref}")
    for entrypoint, rust_type in OBSERVABILITY_COMPONENT_TYPES.items():
        schema = schemas.get(entrypoint, {})
        expected_ref = f"{OBSERVABILITY_SCHEMA_PATH.name}#/$defs/{entrypoint}"
        if schema.get("$ref") != expected_ref or schema.get("x-kiln-field-schema-status") != "complete":
            errors.append(f"observability component {entrypoint} must use complete ref {expected_ref}")
        if schema.get("x-kiln-rust-type") != rust_type:
            errors.append(f"observability component {entrypoint} must bind Rust type {rust_type}")

    for reference in collect_references(document):
        try:
            resolve_contract_ref(document, reference)
        except ContractError as error:
            errors.append(str(error))
    return errors


def collect_references(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "$ref" and isinstance(child, str):
                yield child
            else:
                yield from collect_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from collect_references(child)


def validate_inference_schema(
    schema: dict[str, Any], thinking_schema: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    expected_identity = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://ericflo.github.io/kiln/contracts/kiln-inference-v1.schema.json",
        "x-kiln-field-schema-status": "complete",
    }
    for key, expected in expected_identity.items():
        if schema.get(key) != expected:
            errors.append(f"inference schema {key} must be {expected!r}")
    if schema.get("x-kiln-entrypoints") != list(INFERENCE_ENTRYPOINTS):
        errors.append("inference schema entrypoints drifted")
    if schema.get("oneOf") != [
        {"$ref": f"#/$defs/{name}"}
        for name in INFERENCE_ENTRYPOINTS
        if name != "ChatCompletionChunkStream"
    ]:
        errors.append("inference schema root union must contain the six JSON entrypoints")

    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        errors.append("inference schema $defs must be an object")
        return errors
    if len(definitions) != 52:
        errors.append(f"inference schema must contain 52 definitions, got {len(definitions)}")
    if list(definitions) != sorted(definitions):
        errors.append("inference schema definitions must be sorted")

    expected_rust_types = {
        "BatchCompletionRequest": "BatchCompletionRequest",
        "BatchCompletionResponse": "BatchCompletionResponse",
        "ChatCompletionChunkStream": "ChatCompletionChunkStream",
        "ChatCompletionRequest": "ChatCompletionRequest",
        "ChatCompletionResponse": "ChatCompletionResponse",
        "TextCompletionRequest": "TextCompletionRequest",
        "TextCompletionResponse": "TextCompletionResponse",
    }
    for name, rust_type in expected_rust_types.items():
        definition = definitions.get(name, {})
        if definition.get("x-kiln-rust-type") != rust_type:
            errors.append(f"inference definition {name} must bind Rust type {rust_type}")
        if definition.get("x-kiln-field-schema-status") != "complete":
            errors.append(f"inference definition {name} must be field-schema complete")

    for name in ("BatchCompletionRequest", "ChatCompletionRequest", "TextCompletionRequest"):
        definition = definitions.get(name, {})
        if definition.get("additionalProperties") is not True:
            errors.append(f"{name} must expose its runtime-compatible open input policy")
        if definition.get("x-kiln-unknown-field-policy") != "accepted_and_ignored":
            errors.append(f"{name} must name the accepted-and-ignored unknown-field policy")
    for name in ("BatchCompletionResponse", "ChatCompletionResponse", "TextCompletionResponse"):
        if definitions.get(name, {}).get("additionalProperties") is not False:
            errors.append(f"{name} must be a closed emitted response schema")
    stream = definitions.get("ChatCompletionChunkStream", {})
    if stream.get("type") != "string" or stream.get("contentMediaType") != "text/event-stream":
        errors.append("ChatCompletionChunkStream must describe the SSE transport body")
    if len(stream.get("x-kiln-event-schemas", [])) != 2:
        errors.append("ChatCompletionChunkStream must enumerate chunk and token-timing events")

    preset = definitions.get("SamplingPreset", {})
    if preset.get("x-kiln-unknown-value-policy") != "fallback_to_qwen3_thinking_general":
        errors.append("SamplingPreset must expose the current unknown-value fallback footgun")
    sampling_number = definitions.get("NullableSamplingNumber", {})
    if sampling_number.get("x-kiln-runtime-validation") != "finite_json_number_only":
        errors.append("sampling-number fields must expose the current range-validation gap")

    registry = {
        THINKING_SCHEMA_PATH.name: thinking_schema,
        thinking_schema.get("$id", ""): thinking_schema,
        schema.get("$id", ""): schema,
    }
    for reference in collect_references(schema):
        try:
            resolve_ref({"$ref": reference}, schema, registry)
        except SchemaResolutionError as error:
            errors.append(str(error))

    reachable: set[str] = set()
    pending = list(INFERENCE_ENTRYPOINTS)
    while pending:
        name = pending.pop()
        if name in reachable or name not in definitions:
            continue
        reachable.add(name)
        for reference in collect_references(definitions[name]):
            match = re.fullmatch(r"#/\$defs/([^/]+)", reference)
            if match:
                pending.append(match.group(1))
    orphaned = sorted(set(definitions) - reachable)
    if orphaned:
        errors.append("inference schema has unreachable definitions: " + ", ".join(orphaned))

    examples = schema.get("x-kiln-examples")
    expected_examples = {
        "BatchCompletionRequest",
        "BatchCompletionResponse",
        "ChatCompletionChunk",
        "ChatCompletionRequest",
        "ChatCompletionResponse",
        "RolloutProvenanceV1",
        "StreamingTokenTiming",
        "TextCompletionRequest",
        "TextCompletionResponse",
    }
    if not isinstance(examples, dict) or set(examples) != expected_examples:
        errors.append("inference schema examples must cover every public JSON shape and stream event")
    else:
        for name, values in examples.items():
            if not isinstance(values, list) or not values:
                errors.append(f"inference examples for {name} must be a non-empty array")
                continue
            for index, value in enumerate(values):
                errors.extend(
                    validate_instance(
                        value,
                        {"$ref": f"#/$defs/{name}"},
                        schema,
                        f"example({name})[{index}]",
                        registry=registry,
                    )
                )
    return errors


def validate_observability_schema(schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_identity = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://ericflo.github.io/kiln/contracts/kiln-observability-v1.schema.json",
        "x-kiln-field-schema-status": "complete",
    }
    for key, expected in expected_identity.items():
        if schema.get(key) != expected:
            errors.append(f"observability schema {key} must be {expected!r}")
    if schema.get("x-kiln-entrypoints") != list(OBSERVABILITY_ENTRYPOINTS):
        errors.append("observability schema entrypoints drifted")
    if schema.get("oneOf") != [
        {"$ref": f"#/$defs/{name}"} for name in OBSERVABILITY_ENTRYPOINTS
    ]:
        errors.append("observability schema root union must contain every public response shape")

    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        errors.append("observability schema $defs must be an object")
        return errors
    if len(definitions) != 135:
        errors.append(f"observability schema must contain 135 definitions, got {len(definitions)}")
    if list(definitions) != sorted(definitions):
        errors.append("observability schema definitions must be sorted")
    for name, definition in definitions.items():
        if not isinstance(definition, dict):
            errors.append(f"observability definition {name} must be an object")
            continue
        if definition.get("x-kiln-field-schema-status") != "complete":
            errors.append(f"observability definition {name} must be field-schema complete")
        if not isinstance(definition.get("x-kiln-rust-type"), str):
            errors.append(f"observability definition {name} must bind a Rust wire type")
        if definition.get("type") == "object" and definition.get("additionalProperties") is not False:
            errors.append(f"observability object definition {name} must be closed")

    for entrypoint in OBSERVABILITY_ENTRYPOINTS:
        if entrypoint not in definitions:
            errors.append(f"observability schema is missing entrypoint {entrypoint}")

    for reference in collect_references(schema):
        try:
            resolve_ref({"$ref": reference}, schema)
        except SchemaResolutionError as error:
            errors.append(str(error))

    reachable: set[str] = set()
    pending = list(OBSERVABILITY_ENTRYPOINTS)
    while pending:
        name = pending.pop()
        if name in reachable or name not in definitions:
            continue
        reachable.add(name)
        for reference in collect_references(definitions[name]):
            match = re.fullmatch(r"#/\$defs/([^/]+)", reference)
            if match:
                pending.append(match.group(1))
    orphaned = sorted(set(definitions) - reachable)
    if orphaned:
        errors.append("observability schema has unreachable definitions: " + ", ".join(orphaned))

    examples = schema.get("x-kiln-examples")
    if not isinstance(examples, dict) or set(examples) != set(OBSERVABILITY_ENTRYPOINTS):
        errors.append("observability examples must cover every public JSON response shape")
    else:
        for name, values in examples.items():
            if not isinstance(values, list) or not values:
                errors.append(f"observability examples for {name} must be a non-empty array")
                continue
            for index, value in enumerate(values):
                errors.extend(
                    validate_instance(
                        value,
                        {"$ref": f"#/$defs/{name}"},
                        schema,
                        f"example({name})[{index}]",
                    )
                )

    health = definitions.get("HealthResponse", {})
    status = health.get("properties", {}).get("status", {})
    if status.get("enum") != ["ok", "degraded", "maintenance"]:
        errors.append("HealthResponse must publish the complete readiness status enum")
    if "self_improve_scheduler" in health.get("required", []):
        errors.append("HealthResponse must preserve skipped-when-unarmed scheduler semantics")
    for name in ("ConfigResponse", "HealthResponse", "ModelStateResponse", "ModelsResponse"):
        if definitions.get(name, {}).get("additionalProperties") is not False:
            errors.append(f"{name} must remain a closed emitted response")
    request_record = definitions.get("RequestRecord", {})
    optional_request_fields = {
        "adapter", "temperature", "top_p", "max_tokens", "ttft_ms", "model_prefill_ms",
        "model_decode_ms", "error", "thinking_mode", "prefix_cache", "prompt_full",
        "completion_full", "user_agent", "client", "thinking_budget",
    }
    if optional_request_fields & set(request_record.get("required", [])):
        errors.append("RequestRecord must preserve serde skipped-option wire semantics")
    return errors


def run_inference_self_tests(
    schema: dict[str, Any], thinking_schema: dict[str, Any]
) -> list[str]:
    examples = schema["x-kiln-examples"]
    cases: list[tuple[str, dict[str, Any], str]] = []

    cases.append(("ChatCompletionRequest", {}, "missing messages"))
    chat_stream_n = copy.deepcopy(examples["ChatCompletionRequest"][0])
    chat_stream_n.update({"stream": True, "n": 2})
    cases.append(("ChatCompletionRequest", chat_stream_n, "streaming multiple choices"))
    chat_adapters = copy.deepcopy(examples["ChatCompletionRequest"][0])
    chat_adapters.update({"adapter": "one", "adapters": [{"name": "two", "scale": 1}]})
    cases.append(("ChatCompletionRequest", chat_adapters, "adapter exclusivity"))
    cases.append(("BatchCompletionRequest", {"prompts": []}, "empty batch"))
    cases.append(("TextCompletionRequest", {"prompt": [1]}, "missing prompt_logprobs"))
    cases.append(
        (
            "TextCompletionRequest",
            {"prompt": [1], "prompt_logprobs": 1, "max_tokens": 2},
            "generation-only text completion",
        )
    )
    response_extra = copy.deepcopy(examples["ChatCompletionResponse"][0])
    response_extra["unknown"] = True
    cases.append(("ChatCompletionResponse", response_extra, "unknown response field"))
    bad_rollout = copy.deepcopy(examples["RolloutProvenanceV1"][0])
    bad_rollout["action_tokens"][0]["behavior_logprob"] = None
    cases.append(("RolloutProvenanceV1", bad_rollout, "sampled action probability"))

    registry = {
        THINKING_SCHEMA_PATH.name: thinking_schema,
        thinking_schema.get("$id", ""): thinking_schema,
    }
    errors = []
    for name, value, label in cases:
        observed = validate_instance(
            value, {"$ref": f"#/$defs/{name}"}, schema, registry=registry
        )
        if not observed:
            errors.append(f"inference self-test {label!r} unexpectedly passed")
    return errors


def run_observability_self_tests(schema: dict[str, Any]) -> list[str]:
    examples = schema["x-kiln-examples"]
    cases: list[tuple[str, Any, str]] = []
    health_missing = copy.deepcopy(examples["HealthResponse"][0])
    health_missing.pop("status")
    cases.append(("HealthResponse", health_missing, "missing health status"))
    health_extra = copy.deepcopy(examples["HealthResponse"][0])
    health_extra["unknown"] = True
    cases.append(("HealthResponse", health_extra, "unknown health field"))
    models_owner = copy.deepcopy(examples["ModelsResponse"][0])
    models_owner["data"][0]["owned_by"] = "other"
    cases.append(("ModelsResponse", models_owner, "model ownership constant"))
    negative_decode = copy.deepcopy(examples["DecodeStatsSnapshot"][0])
    negative_decode["p99_itl_ms"] = -1
    cases.append(("DecodeStatsSnapshot", negative_decode, "negative decode latency"))
    nullable_skipped = copy.deepcopy(examples["RequestRecord"][0])
    nullable_skipped["adapter"] = None
    cases.append(("RequestRecord", nullable_skipped, "null skipped request field"))
    debug_missing = copy.deepcopy(examples["DebugProvenanceErrorResponse"][0])
    debug_missing.pop("detail")
    cases.append(("DebugProvenanceErrorResponse", debug_missing, "missing debug error detail"))
    cache_negative = copy.deepcopy(examples["CacheStatsResponse"][0])
    cache_negative["stats"]["total_entries"] = -1
    cases.append(("CacheStatsResponse", cache_negative, "negative cache entry count"))

    errors = []
    for name, value, label in cases:
        observed = validate_instance(value, {"$ref": f"#/$defs/{name}"}, schema)
        if not observed:
            errors.append(f"observability self-test {label!r} unexpectedly passed")

    open_health = copy.deepcopy(schema)
    open_health["$defs"]["HealthResponse"]["additionalProperties"] = True
    if not any("HealthResponse must be closed" in error for error in validate_observability_schema(open_health)):
        errors.append("observability self-test failed to reject an open HealthResponse")
    return errors


def run_self_tests(
    document: dict[str, Any], inference_schema: dict[str, Any], observability_schema: dict[str, Any],
    thinking_schema: dict[str, Any]
) -> list[str]:
    mutations = []

    duplicate = copy.deepcopy(document)
    duplicate["paths"]["/v1/models"]["get"]["operationId"] = duplicate["paths"]["/health"]["get"]["operationId"]
    mutations.append((duplicate, "duplicate operationId"))

    missing_path_parameter = copy.deepcopy(document)
    missing_path_parameter["paths"]["/v1/adapters/{name}"]["delete"].pop("parameters")
    mutations.append((missing_path_parameter, "path parameters"))

    missing_body = copy.deepcopy(document)
    missing_body["paths"]["/v1/chat/completions"]["post"].pop("requestBody")
    mutations.append((missing_body, "requestBody presence"))

    bad_media = copy.deepcopy(document)
    response = bad_media["paths"]["/metrics"]["get"]["responses"]["200"]
    response["content"] = {"application/x-unknown": next(iter(response["content"].values()))}
    mutations.append((bad_media, "unsupported"))

    bad_ref = copy.deepcopy(document)
    bad_ref["paths"]["/v1/models"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"] = "#/components/schemas/Missing"
    mutations.append((bad_ref, "unresolved OpenAPI reference"))

    errors = []
    for mutated, expected_fragment in mutations:
        observed = validate_contract(mutated)
        if not any(expected_fragment in error for error in observed):
            errors.append(f"self-test mutation did not produce {expected_fragment!r}: {observed[:3]}")
    errors.extend(run_inference_self_tests(inference_schema, thinking_schema))
    errors.extend(run_observability_self_tests(observability_schema))
    return errors


def check(*, self_test: bool) -> None:
    document = load_contract()
    inference_schema = load_json(INFERENCE_SCHEMA_PATH)
    observability_schema = load_json(OBSERVABILITY_SCHEMA_PATH)
    thinking_schema = load_json(THINKING_SCHEMA_PATH)
    errors = validate_contract(document)
    errors.extend(validate_inference_schema(inference_schema, thinking_schema))
    errors.extend(validate_observability_schema(observability_schema))
    if self_test:
        errors.extend(run_self_tests(document, inference_schema, observability_schema, thinking_schema))
    if errors:
        raise ContractError("HTTP API contract failed:\n- " + "\n- ".join(errors))
    print(
        "HTTP API contract passed: "
        f"{document['x-kiln-path-count']} paths, "
        f"{document['x-kiln-operation-count']} operations "
        f"({document['x-kiln-method-counts']}), "
        f"{len(document['components']['schemas'])} payload components "
        f"({document['x-kiln-component-schema-counts']['complete']} complete, "
        f"{document['x-kiln-component-schema-counts']['migration_pending']} migration pending), "
        f"{len(inference_schema['$defs'])} inference definitions, "
        f"{len(observability_schema['$defs'])} observability definitions"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="also run fail-closed mutation cases")
    args = parser.parse_args()
    try:
        check(self_test=args.self_test)
    except ContractError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
