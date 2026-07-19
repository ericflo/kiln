"""Closed request and evidence contracts for an HF next-token oracle."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

from strict_json import loads as strict_json_loads


REQUEST_SCHEMA = "kiln.hf-next-token-request.v1"
PASS_PREFIX = "KILN_HF_NEXT_TOKEN_REFERENCE_PASS "
VOCAB_SIZE = 248_320
MAX_REQUEST_BYTES = 256 * 1024


class ContractError(RuntimeError):
    """A next-token request or result is not the exact declared contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_bytes(value)).hexdigest()}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return value


def _exact(value: dict[str, Any], fields: set[str], label: str) -> None:
    actual = set(value)
    if actual != fields:
        raise ContractError(
            f"{label} fields are not closed: missing={sorted(fields - actual)}, "
            f"unexpected={sorted(actual - fields)}"
        )


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a nonempty string")
    return value


def _sha256(value: Any, label: str) -> str:
    value = _string(value, label)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ContractError(f"{label} must be a canonical sha256")
    return value


def _token_id(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < VOCAB_SIZE:
        raise ContractError(f"{label} must be an integer in 0..{VOCAB_SIZE - 1}")
    return value


def _token_ids(value: Any, label: str, *, maximum: int = 4096) -> list[int]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise ContractError(f"{label} must contain 1..{maximum} token IDs")
    return [_token_id(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _receipt_reference(value: Any, label: str) -> dict[str, Any]:
    value = _object(value, label)
    _exact(value, {"content_sha256", "file_sha256", "path"}, label)
    path = _string(value["path"], f"{label}.path")
    normalized = PurePosixPath(path)
    if normalized.is_absolute() or ".." in normalized.parts or str(normalized) != path:
        raise ContractError(f"{label}.path must be a normalized repository-relative path")
    _sha256(value["file_sha256"], f"{label}.file_sha256")
    _sha256(value["content_sha256"], f"{label}.content_sha256")
    return value


def validate_request(value: Any) -> dict[str, Any]:
    value = _object(value, "request")
    _exact(
        value,
        {
            "candidates",
            "continuation_prefix",
            "id",
            "input_token_ids",
            "input_token_ids_sha256",
            "model_identity",
            "prompt",
            "schema",
            "source",
        },
        "request",
    )
    if value["schema"] != REQUEST_SCHEMA:
        raise ContractError(f"request.schema must equal {REQUEST_SCHEMA}")
    request_id = _string(value["id"], "request.id")
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", request_id) is None:
        raise ContractError("request.id must be a portable identifier")

    source = _object(value["source"], "request.source")
    _exact(
        source,
        {"kiln_receipt", "source_commit", "vllm_receipt"},
        "request.source",
    )
    commit = _string(source["source_commit"], "request.source.source_commit")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ContractError("request.source.source_commit must be a full Git commit")
    _receipt_reference(source["kiln_receipt"], "request.source.kiln_receipt")
    _receipt_reference(source["vllm_receipt"], "request.source.vllm_receipt")

    model = _object(value["model_identity"], "request.model_identity")
    _exact(
        model,
        {
            "chat_template_hash",
            "config_hash",
            "content_sha256",
            "id",
            "tokenizer_hash",
            "weight_files",
        },
        "request.model_identity",
    )
    _string(model["id"], "request.model_identity.id")
    for name in ("chat_template_hash", "config_hash", "content_sha256", "tokenizer_hash"):
        _sha256(model[name], f"request.model_identity.{name}")
    weights = model["weight_files"]
    if not isinstance(weights, list) or not weights:
        raise ContractError("request.model_identity.weight_files must be nonempty")
    weight_paths: list[str] = []
    for index, item in enumerate(weights):
        item = _object(item, f"request.model_identity.weight_files[{index}]")
        _exact(item, {"bytes", "path", "sha256"}, f"request.model_identity.weight_files[{index}]")
        path = _string(item["path"], f"request.model_identity.weight_files[{index}].path")
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise ContractError("model weight paths must be repository-relative")
        if (
            isinstance(item["bytes"], bool)
            or not isinstance(item["bytes"], int)
            or item["bytes"] <= 0
        ):
            raise ContractError("model weight byte counts must be positive integers")
        _sha256(item["sha256"], f"request.model_identity.weight_files[{index}].sha256")
        weight_paths.append(path)
    if weight_paths != sorted(set(weight_paths)):
        raise ContractError("model weight paths must be sorted and unique")

    prompt = _object(value["prompt"], "request.prompt")
    _exact(
        prompt,
        {
            "add_generation_prompt",
            "generator",
            "messages",
            "template_kwargs",
            "token_count",
            "token_ids",
            "utf8_sha256",
        },
        "request.prompt",
    )
    if prompt["add_generation_prompt"] is not True:
        raise ContractError("request.prompt.add_generation_prompt must be true")
    messages = prompt["messages"]
    if not isinstance(messages, list) or len(messages) != 1:
        raise ContractError("request.prompt.messages must contain one user message")
    message = _object(messages[0], "request.prompt.messages[0]")
    _exact(message, {"content", "role"}, "request.prompt.messages[0]")
    if message["role"] != "user":
        raise ContractError("request.prompt.messages[0].role must equal user")
    content = _string(message["content"], "request.prompt.messages[0].content")
    expected_utf8 = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
    if _sha256(prompt["utf8_sha256"], "request.prompt.utf8_sha256") != expected_utf8:
        raise ContractError("request.prompt.utf8_sha256 does not match message content")
    template_kwargs = _object(prompt["template_kwargs"], "request.prompt.template_kwargs")
    _exact(template_kwargs, {"enable_thinking"}, "request.prompt.template_kwargs")
    if template_kwargs["enable_thinking"] is not False:
        raise ContractError("request.prompt.template_kwargs.enable_thinking must be false")
    generator = _object(prompt["generator"], "request.prompt.generator")
    _exact(
        generator,
        {"phase", "profile", "request_index", "run_id", "template_version"},
        "request.prompt.generator",
    )
    for name in ("phase", "profile", "run_id", "template_version"):
        _string(generator[name], f"request.prompt.generator.{name}")
    if (
        generator["profile"] != "short"
        or generator["template_version"] != "fixed-serving-profiles-v1"
    ):
        raise ContractError("request prompt generator must use the retained short serving profile")
    if isinstance(generator["request_index"], bool) or generator["request_index"] != 0:
        raise ContractError("request.prompt.generator.request_index must equal zero")
    prompt_ids = _token_ids(prompt["token_ids"], "request.prompt.token_ids")
    if isinstance(prompt["token_count"], bool) or prompt["token_count"] != len(prompt_ids):
        raise ContractError("request.prompt.token_count does not match token_ids")

    prefix = value["continuation_prefix"]
    if not isinstance(prefix, list) or not 1 <= len(prefix) <= 64:
        raise ContractError("request.continuation_prefix must contain 1..64 tokens")
    prefix_ids: list[int] = []
    for index, item in enumerate(prefix):
        item = _object(item, f"request.continuation_prefix[{index}]")
        _exact(item, {"text", "token_id"}, f"request.continuation_prefix[{index}]")
        prefix_ids.append(
            _token_id(
                item["token_id"],
                f"request.continuation_prefix[{index}].token_id",
            )
        )
        if not isinstance(item["text"], str):
            raise ContractError(f"request.continuation_prefix[{index}].text must be a string")

    input_ids = _token_ids(value["input_token_ids"], "request.input_token_ids")
    if input_ids != prompt_ids + prefix_ids:
        raise ContractError("request.input_token_ids must equal prompt plus continuation prefix")
    if _sha256(
        value["input_token_ids_sha256"], "request.input_token_ids_sha256"
    ) != canonical_sha256(input_ids):
        raise ContractError("request.input_token_ids_sha256 does not match input_token_ids")

    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ContractError("request.candidates must contain the Kiln and vLLM tokens")
    engines: list[str] = []
    candidate_ids: list[int] = []
    for index, item in enumerate(candidates):
        item = _object(item, f"request.candidates[{index}]")
        _exact(item, {"engine", "text", "token_id"}, f"request.candidates[{index}]")
        engines.append(_string(item["engine"], f"request.candidates[{index}].engine"))
        candidate_ids.append(_token_id(item["token_id"], f"request.candidates[{index}].token_id"))
        if not isinstance(item["text"], str):
            raise ContractError(f"request.candidates[{index}].text must be a string")
    if engines != ["kiln", "vllm"] or len(set(candidate_ids)) != 2:
        raise ContractError("request candidates must be distinct and ordered kiln, vllm")
    return value


def load_request(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"next-token request is not a regular file: {path}")
    payload = path.read_bytes()
    if len(payload) > MAX_REQUEST_BYTES:
        raise ContractError(f"next-token request exceeds {MAX_REQUEST_BYTES} bytes")
    try:
        value = strict_json_loads(payload)
    except Exception as exc:
        raise ContractError(f"cannot parse next-token request {path}: {exc}") from exc
    return validate_request(value), canonical_sha256(value)


def validate_source_receipts(request: dict[str, Any], root: Path) -> None:
    for engine in ("kiln", "vllm"):
        reference = request["source"][f"{engine}_receipt"]
        path = root / reference["path"]
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"{engine} source receipt is not a regular file: {path}")
        if file_sha256(path) != reference["file_sha256"]:
            raise ContractError(f"{engine} source receipt file hash changed")
        try:
            receipt = strict_json_loads(path.read_bytes())
        except Exception as exc:
            raise ContractError(f"cannot parse {engine} source receipt: {exc}") from exc
        if (
            not isinstance(receipt, dict)
            or receipt.get("receipt_sha256") != reference["content_sha256"]
        ):
            raise ContractError(f"{engine} source receipt content hash changed")
        repository = receipt.get("driver_environment", {}).get("repository", {})
        if (
            receipt.get("driver_version") != "7"
            or repository.get("commit") != request["source"]["source_commit"]
            or repository.get("dirty") is not False
        ):
            raise ContractError(f"{engine} source receipt source identity changed")
        if (
            receipt.get("workload", {}).get("run_id")
            != request["prompt"]["generator"]["run_id"]
        ):
            raise ContractError(f"{engine} source receipt run ID changed")
        expected_identity = dict(request["model_identity"])
        actual_identity = dict(receipt.get("engine", {}).get("model_identity", {}))
        actual_identity.pop("path", None)
        if actual_identity != expected_identity:
            raise ContractError(f"{engine} source receipt model identity changed")
        runs = receipt.get("runs")
        if not isinstance(runs, list) or len(runs) != 1:
            raise ContractError(f"{engine} source receipt must contain one measured run")
        run = runs[0]
        if (
            not isinstance(run, dict)
            or run.get("concurrency") != 1
            or run.get("repeat") != 0
            or run.get("prompt_token_counts")
            != [len(request["prompt"]["token_ids"])]
        ):
            raise ContractError(f"{engine} source receipt measured-run identity changed")
        output = run.get("output_evidence")
        if not isinstance(output, list) or len(output) != 1:
            raise ContractError(f"{engine} source receipt must contain one output row")
        row = output[0]
        exact = row.get("exact_output") if isinstance(row, dict) else None
        if not isinstance(exact, dict) or set(exact) != {
            "content_base64",
            "reasoning_content_base64",
        }:
            raise ContractError(f"{engine} source receipt must retain exact output")
        try:
            content = base64.b64decode(
                exact["content_base64"], validate=True
            ).decode("utf-8")
            reasoning = base64.b64decode(
                exact["reasoning_content_base64"], validate=True
            ).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, TypeError) as exc:
            raise ContractError(
                f"{engine} source receipt exact output is invalid: {exc}"
            ) from exc
        candidate = next(
            item for item in request["candidates"] if item["engine"] == engine
        )
        expected_prefix = "".join(
            item["text"] for item in request["continuation_prefix"]
        ) + candidate["text"]
        if reasoning != "" or not content.startswith(expected_prefix):
            raise ContractError(
                f"{engine} source receipt does not begin with the declared token divergence"
            )


def validate_evidence(value: Any) -> dict[str, Any]:
    value = _object(value, "HF next-token evidence")
    required = {
        "attention_implementation",
        "argmax",
        "argmax_text",
        "candidate_tokens",
        "configuration_sha256",
        "deterministic_algorithms",
        "device",
        "dtype",
        "duration_seconds",
        "input_token_count",
        "input_token_ids_sha256",
        "logits_sha256",
        "linear_attention_implementation",
        "memory_high_events",
        "memory_max_events",
        "memory_oom_events",
        "memory_oom_kill_events",
        "memory_peak_bytes",
        "memory_swap_bytes",
        "modeling_sha256",
        "output_bytes",
        "request_id",
        "request_sha256",
        "tf32_allowed",
        "top_logit_margin",
        "top_tokens",
        "torch_hip_version",
        "torch_commit",
        "torch_version",
        "transformers_version",
        "vocab",
    }
    _exact(value, required, "HF next-token evidence")
    if not isinstance(value["top_tokens"], list) or len(value["top_tokens"]) != 10:
        raise ContractError("HF next-token evidence must retain exactly ten top tokens")
    if not isinstance(value["candidate_tokens"], list) or len(value["candidate_tokens"]) != 2:
        raise ContractError("HF next-token evidence must retain two candidates")
    integer_fields = (
        "argmax",
        "input_token_count",
        "memory_high_events",
        "memory_max_events",
        "memory_oom_events",
        "memory_oom_kill_events",
        "memory_peak_bytes",
        "memory_swap_bytes",
        "output_bytes",
        "vocab",
    )
    for name in integer_fields:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ContractError(f"HF next-token evidence {name} must be nonnegative integer")
    if value["vocab"] != VOCAB_SIZE or value["argmax"] != value["top_tokens"][0].get("token_id"):
        raise ContractError("HF next-token evidence has inconsistent vocabulary or argmax")
    for name in ("duration_seconds", "top_logit_margin"):
        item = value[name]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item <= 0
        ):
            raise ContractError(f"HF next-token evidence {name} must be positive and finite")
    for name in ("logits_sha256", "input_token_ids_sha256", "request_sha256"):
        _sha256(value[name], f"HF next-token evidence.{name}")
    for name in ("configuration_sha256", "modeling_sha256"):
        _sha256(value[name], f"HF next-token evidence.{name}")
    for name in (
        "argmax_text",
        "device",
        "request_id",
        "torch_hip_version",
        "torch_commit",
        "torch_version",
        "transformers_version",
    ):
        _string(value[name], f"HF next-token evidence.{name}")
    expected_routes = {
        "attention_implementation": "eager",
        "configuration_sha256": "sha256:3c01b3cdcff8d77cbafac9841bc48c41e5a5b38637231f1bde3d843cd198dbaf",
        "deterministic_algorithms": True,
        "dtype": "bfloat16",
        "linear_attention_implementation": "transformers_torch_fallback",
        "modeling_sha256": "sha256:cf085792cb59e5bdf9b88a3d20bd353892289d054662a9c2b662221b97caefba",
        "tf32_allowed": False,
        "torch_commit": "cf30153c4c131c8164ee7798e5022d810682e2cb",
        "torch_version": "2.13.0+rocm7.2",
        "transformers_version": "5.13.1",
    }
    if any(value[name] != expected for name, expected in expected_routes.items()):
        raise ContractError("HF next-token evidence does not use the pinned eager routes")
    top_ids: list[int] = []
    previous_logit = math.inf
    for index, item in enumerate(value["top_tokens"]):
        item = _object(item, f"HF next-token evidence.top_tokens[{index}]")
        _exact(item, {"logit", "text", "token_id"}, f"HF next-token evidence.top_tokens[{index}]")
        token_id = _token_id(
            item["token_id"],
            f"HF next-token evidence.top_tokens[{index}].token_id",
        )
        if not isinstance(item["text"], str):
            raise ContractError("HF next-token top-token text must be a string")
        logit = item["logit"]
        if (
            isinstance(logit, bool)
            or not isinstance(logit, (int, float))
            or not math.isfinite(logit)
        ):
            raise ContractError("HF next-token top-token logits must be finite")
        if logit > previous_logit:
            raise ContractError("HF next-token top tokens must be sorted by descending logit")
        previous_logit = logit
        top_ids.append(token_id)
    if len(set(top_ids)) != 10:
        raise ContractError("HF next-token top-token IDs must be unique")
    if value["argmax_text"] != value["top_tokens"][0]["text"]:
        raise ContractError("HF next-token argmax text must match the first top token")
    candidate_engines: list[str] = []
    for index, item in enumerate(value["candidate_tokens"]):
        item = _object(item, f"HF next-token evidence.candidate_tokens[{index}]")
        _exact(
            item,
            {"engine", "logit", "rank", "text", "token_id"},
            f"HF next-token evidence.candidate_tokens[{index}]",
        )
        candidate_engines.append(_string(item["engine"], f"candidate[{index}].engine"))
        _token_id(item["token_id"], f"candidate[{index}].token_id")
        if not isinstance(item["text"], str):
            raise ContractError("HF next-token candidate text must be a string")
        if (
            isinstance(item["rank"], bool)
            or not isinstance(item["rank"], int)
            or not 1 <= item["rank"] <= VOCAB_SIZE
        ):
            raise ContractError("HF next-token candidate rank must be in the vocabulary")
        if (
            isinstance(item["logit"], bool)
            or not isinstance(item["logit"], (int, float))
            or not math.isfinite(item["logit"])
        ):
            raise ContractError("HF next-token candidate logit must be finite")
    if candidate_engines != ["kiln", "vllm"]:
        raise ContractError("HF next-token candidate evidence must be ordered kiln, vllm")
    return value


def parse_pass_marker(output: str) -> dict[str, Any]:
    records = [
        line[len(PASS_PREFIX) :]
        for line in output.splitlines()
        if line.startswith(PASS_PREFIX)
    ]
    if len(records) != 1:
        raise ContractError(f"expected one HF next-token marker, found {len(records)}")
    try:
        value = strict_json_loads(records[0])
    except Exception as exc:
        raise ContractError(f"HF next-token marker is invalid JSON: {exc}") from exc
    return validate_evidence(value)
