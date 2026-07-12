#!/usr/bin/env python3
"""Train one Kiln SFT or recorded-rollout GRPO handoff with HF/TRL/PEFT.

The script deliberately validates the complete Kiln export and the local base
model before importing torch. A successful run writes PEFT artifacts and a
self-verifying ``kiln_hf_result.json`` into the working copy of the bundle.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import stat
import struct
import sys
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any


EXPORT_MANIFEST = "kiln_hf_export.json"
RESULT_MANIFEST = "kiln_hf_result.json"
EXECUTED_SCRIPT = "executed_train.py"
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_MODEL = "adapter_model.safetensors"
RESULT_SENTINEL = ".kiln_hf_result.incomplete"
SHA256_PREFIX = "sha256:"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
GRPO_CORPUS_DOMAIN = b"kiln.hf-trl-grpo-corpus.v1\0"
GRPO_PROVENANCE_SCHEMA = "kiln.rollout-provenance.v1"
MAX_GRPO_DATASET_BYTES = 64 * 1024 * 1024 * 1024
MAX_GRPO_ROW_BYTES = 256 * 1024 * 1024
MAX_GRPO_GROUPS = 10_000_000
MAX_GRPO_COMPLETIONS = 1024
MAX_GRPO_TOKEN_COUNT = 16_777_216
MAX_GRPO_STOP_SEQUENCES = 256
MAX_GRPO_STOP_BYTES = 16 * 1024
MAX_GRPO_TEMPLATE_TOOLS = 256
MAX_GRPO_TEMPLATE_KWARGS = 256
MAX_GRPO_TEMPLATE_BYTES = 1024 * 1024
KILN_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

PINNED_PACKAGES = {
    "accelerate": "1.14.0",
    "datasets": "5.0.0",
    "jinja2": "3.1.6",
    "peft": "0.19.1",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.13.0",
    "transformers": "5.13.1",
    "trl": "1.8.0",
}


class ContractError(RuntimeError):
    """The handoff cannot be trained without weakening its contract."""


def _target_modules(value: str | None) -> list[str]:
    modules = list(KILN_TARGET_MODULES) if value is None else [
        module.strip() for module in value.split(",")
    ]
    if not modules or any(not module for module in modules):
        raise ContractError("target modules must be a non-empty comma-separated list")
    if len(set(modules)) != len(modules):
        raise ContractError("target modules must not contain duplicates")
    unsupported = sorted(set(modules) - set(KILN_TARGET_MODULES))
    if unsupported:
        raise ContractError(
            f"target modules {unsupported!r} are not loadable by Kiln; "
            f"supported modules are {list(KILN_TARGET_MODULES)!r}"
        )
    return modules


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _read_json(path: Path, *, bounded: bool = False) -> Any:
    try:
        size = path.stat().st_size
        if bounded and size > MAX_MANIFEST_BYTES:
            raise ContractError(f"{path} exceeds {MAX_MANIFEST_BYTES} bytes")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read strict JSON from {path}: {exc}") from exc


def _expect_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _expect_keys(
    value: Any,
    label: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    obj = _expect_object(value, label)
    allowed = required | (optional or set())
    missing = sorted(required - obj.keys())
    unknown = sorted(obj.keys() - allowed)
    if missing or unknown:
        raise ContractError(
            f"{label} fields differ: missing={missing!r}, unknown={unknown!r}"
        )
    return obj


def _sha256_bytes(data: bytes) -> str:
    return SHA256_PREFIX + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ContractError(f"cannot hash {path}: {exc}") from exc
    return SHA256_PREFIX + digest.hexdigest()


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith(SHA256_PREFIX):
        raise ContractError(f"{label} must use sha256:<64 lowercase hex>")
    suffix = value[len(SHA256_PREFIX) :]
    if len(suffix) != 64 or any(ch not in "0123456789abcdef" for ch in suffix):
        raise ContractError(f"{label} must use sha256:<64 lowercase hex>")
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ContractError(f"cannot canonicalize manifest JSON: {exc}") from exc
    return _sha256_bytes(encoded)


def _base_weight_aggregate_sha256(shards: list[dict[str, Any]]) -> str:
    records = sorted(
        (
            bytes.fromhex(
                _validate_sha256(shard["sha256"], "base shard hash")[
                    len(SHA256_PREFIX) :
                ]
            ),
            shard["size_bytes"],
        )
        for shard in shards
    )
    digest = hashlib.sha256()
    digest.update(b"kiln.base-model-content.v1\0")
    digest.update(len(records).to_bytes(8, "little"))
    for shard_digest, size_bytes in records:
        digest.update(size_bytes.to_bytes(8, "little"))
        digest.update(shard_digest)
    return SHA256_PREFIX + digest.hexdigest()


class _F32(float):
    """A value serialized with serde_json's finite-f32 formatting rules."""


def _float_components(text: str) -> tuple[str, str, int]:
    sign = ""
    if text.startswith("-"):
        sign, text = "-", text[1:]
    mantissa, marker, raw_exponent = text.lower().partition("e")
    exponent = int(raw_exponent) if marker else 0
    integer, dot, fraction = mantissa.partition(".")
    digits = (integer + (fraction if dot else "")).lstrip("0") or "0"
    decimal_exponent = len(integer) - 1 + exponent
    if integer == "0":
        leading_fraction_zeros = len(fraction) - len(fraction.lstrip("0"))
        decimal_exponent = -leading_fraction_zeros - 1
    return sign, digits, decimal_exponent


def _serde_float_text(value: float, *, bits: int) -> str:
    if not math.isfinite(value):
        raise ContractError("canonical JSON contains a non-finite number")
    if bits == 32:
        try:
            packed = struct.pack("<f", value)
        except OverflowError as exc:
            raise ContractError("canonical JSON f32 is outside the finite range") from exc
        value = struct.unpack("<f", packed)[0]
        if not math.isfinite(value):
            raise ContractError("canonical JSON f32 is outside the finite range")
        if value == 0.0:
            return "-0.0" if math.copysign(1.0, value) < 0.0 else "0.0"
        text = None
        for precision in range(1, 10):
            candidate = format(value, f".{precision}g")
            try:
                same = struct.pack("<f", float(candidate)) == packed
            except (OverflowError, ValueError):
                same = False
            if same:
                text = candidate
                break
        if text is None:
            raise ContractError("cannot format a finite f32 as canonical JSON")
        fixed_min, fixed_max = -6, 12
    else:
        if value == 0.0:
            return "-0.0" if math.copysign(1.0, value) < 0.0 else "0.0"
        text = repr(value)
        fixed_min, fixed_max = -5, 15

    sign, digits, decimal_exponent = _float_components(text)
    if fixed_min <= decimal_exponent <= fixed_max:
        if decimal_exponent >= len(digits) - 1:
            return sign + digits + "0" * (decimal_exponent + 1 - len(digits)) + ".0"
        if decimal_exponent >= 0:
            split = decimal_exponent + 1
            return sign + digits[:split] + "." + digits[split:]
        return sign + "0." + "0" * (-decimal_exponent - 1) + digits
    mantissa = digits[0] if len(digits) == 1 else digits[0] + "." + digits[1:]
    exponent = f"+{decimal_exponent}" if decimal_exponent >= 0 else str(decimal_exponent)
    return sign + mantissa + "e" + exponent


def _compact_json_bytes(value: Any) -> bytes:
    """Match serde_json's compact representation for admitted JSON values."""

    def encode(item: Any) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if isinstance(item, int):
            return str(item)
        if isinstance(item, _F32):
            return _serde_float_text(item, bits=32)
        if isinstance(item, float):
            return _serde_float_text(item, bits=64)
        if isinstance(item, str):
            return json.dumps(item, ensure_ascii=False, allow_nan=False)
        if isinstance(item, list):
            return "[" + ",".join(encode(element) for element in item) + "]"
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ContractError("canonical JSON object keys must be strings")
            return "{" + ",".join(
                f"{encode(key)}:{encode(element)}" for key, element in item.items()
            ) + "}"
        raise ContractError(f"canonical JSON contains unsupported {type(item).__name__}")

    try:
        return encode(value).encode("utf-8")
    except UnicodeError as exc:
        raise ContractError(f"cannot encode canonical JSON: {exc}") from exc


def _validate_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{label} is not a normalized relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ContractError(f"{label} is not a normalized relative path")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ContractError(f"{label} contains a control character")
    return value


def _require_real_bundle_root(root: Path) -> Path:
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise ContractError(f"cannot stat bundle root {root}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ContractError(f"bundle root must be a real directory: {root}")
    return root.resolve(strict=True)


def _bundle_file(root: Path, relative: str) -> Path:
    relative = _validate_relative_path(relative, "bundle artifact path")
    current = root
    for index, part in enumerate(PurePosixPath(relative).parts):
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ContractError(f"cannot stat bundle artifact {current}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ContractError(f"bundle artifact traverses a symlink: {current}")
        final = index + 1 == len(PurePosixPath(relative).parts)
        if final and not stat.S_ISREG(metadata.st_mode):
            raise ContractError(f"bundle artifact is not a regular file: {current}")
        if not final and not stat.S_ISDIR(metadata.st_mode):
            raise ContractError(f"bundle artifact parent is not a directory: {current}")
    return current


def _verify_identity(root: Path, value: Any, expected_path: str) -> Path:
    identity = _expect_keys(
        value,
        f"identity for {expected_path}",
        {"relative_path", "size_bytes", "sha256"},
    )
    relative = _validate_relative_path(identity["relative_path"], "artifact relative_path")
    if relative != expected_path:
        raise ContractError(f"artifact path {relative!r} must be {expected_path!r}")
    size = identity["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ContractError(f"artifact {relative!r} has invalid size_bytes")
    expected_hash = _validate_sha256(identity["sha256"], f"artifact {relative!r} hash")
    path = _bundle_file(root, relative)
    if path.stat().st_size != size:
        raise ContractError(f"artifact {relative!r} size differs from its manifest")
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        raise ContractError(
            f"artifact {relative!r} hash differs: manifest={expected_hash}, actual={actual_hash}"
        )
    return path


def _validate_source_provenance(value: Any) -> dict[str, Any]:
    provenance = _expect_keys(
        value,
        "source_execution_provenance",
        {
            "schema_version",
            "provenance_type",
            "backend",
            "build",
            "model",
            "precision",
            "kernels",
            "configuration",
            "provenance_sha256",
        },
    )
    if provenance["schema_version"] != 1 or provenance["provenance_type"] != "kiln.execution-provenance.v1":
        raise ContractError("unsupported source execution provenance")
    _expect_keys(provenance["backend"], "source backend", {"name", "device", "numerical_runtime_sha256"})
    _expect_keys(
        provenance["build"],
        "source build",
        {"package_version", "target", "executable_sha256"},
        {"git_commit", "source_tree_sha256", "source_dirty"},
    )
    _expect_keys(
        provenance["model"],
        "source model",
        {"model_config_sha256", "tokenizer_vocab_sha256", "tokenizer_config_sha256"},
        {"chat_template_sha256", "training_chat_template_sha256"},
    )
    _expect_keys(provenance["precision"], "source precision", {"inference_dtype", "training_policy"})
    _expect_keys(
        provenance["kernels"],
        "source kernels",
        {"contract_type", "versions", "compiled_features", "contract_sha256"},
    )
    _expect_keys(
        provenance["configuration"],
        "source configuration",
        {"effective_server_config_sha256", "effective_environment_sha256"},
    )
    _validate_sha256(provenance["provenance_sha256"], "source provenance digest")
    return provenance


def _declared_export_files(root: Path, manifest: dict[str, Any]) -> set[str]:
    model = manifest["model"]
    data = manifest["data"]
    declared = {
        EXPORT_MANIFEST,
        model["model_config"]["relative_path"],
        model["tokenizer"]["relative_path"],
        model["chat_template"]["relative_path"],
        model["native_training_chat_template"]["relative_path"],
        model["trl_training_chat_template"]["relative_path"],
        data["dataset"]["relative_path"],
        manifest["reference_script"]["relative_path"],
        manifest["environment_lock"]["relative_path"],
    }
    if manifest["task"] == "sft":
        declared.add(data["sft_selection"]["ingestion_receipt"]["relative_path"])
    if "split_manifest" in data:
        declared.add(data["split_manifest"]["relative_path"])
    adapter = manifest.get("input_adapter")
    if adapter:
        declared.add(adapter["config"]["relative_path"])
        declared.add(adapter["model"]["relative_path"])
        if "kiln_manifest" in adapter:
            declared.add(adapter["kiln_manifest"]["relative_path"])

    actual: set[str] = set()
    for directory, directories, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directories:
            path = directory_path / name
            if path.is_symlink():
                raise ContractError(f"bundle tree contains a symlink: {path}")
        for name in files:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise ContractError(f"bundle tree contains a non-regular file: {path}")
            actual.add(path.relative_to(root).as_posix())
    if actual != declared:
        raise ContractError(
            f"bundle file set differs: missing={sorted(declared - actual)!r}, "
            f"unexpected={sorted(actual - declared)!r}"
        )
    return declared


def _validate_input_adapter(root: Path, value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    adapter = _expect_keys(
        value,
        "input adapter",
        {"name", "config", "model"},
        {"kiln_manifest"},
    )
    _validate_identity_text(adapter["name"], "input adapter name")
    _verify_identity(root, adapter["config"], "input_adapter/adapter_config.json")
    _verify_identity(root, adapter["model"], "input_adapter/adapter_model.safetensors")
    if "kiln_manifest" in adapter:
        _verify_identity(
            root,
            adapter["kiln_manifest"],
            "input_adapter/adapter_manifest.json",
        )
    return adapter


def _validate_sft_export_data(root: Path, value: Any) -> dict[str, Any]:
    data = _expect_keys(
        value,
        "export data",
        {
            "source_name",
            "format",
            "row_count",
            "ordered_corpus_sha256",
            "dataset",
            "sft_selection",
        },
        {"split_manifest"},
    )
    if (
        data["format"] != "sft_messages_jsonl"
        or not isinstance(data["row_count"], int)
        or isinstance(data["row_count"], bool)
        or data["row_count"] <= 0
    ):
        raise ContractError("invalid SFT data format or row_count")
    selection = _expect_keys(
        data["sft_selection"],
        "SFT selection",
        {
            "invalid_row_policy",
            "label_policy",
            "rows_read",
            "rows_kept",
            "rows_rejected",
            "kept_corpus_sha256",
            "ingestion_receipt",
        },
    )
    if selection["label_policy"] != "assistant_only_generation_spans":
        raise ContractError("SFT export does not require assistant-only generation spans")
    if (
        data["row_count"] != selection["rows_kept"]
        or data["ordered_corpus_sha256"] != selection["kept_corpus_sha256"]
    ):
        raise ContractError("SFT data identity differs from its selection receipt")
    _verify_identity(root, data["dataset"], "train.jsonl")
    ingestion_path = _verify_identity(
        root, selection["ingestion_receipt"], "sft_ingestion.json"
    )
    ingestion = _expect_keys(
        _read_json(ingestion_path),
        "SFT ingestion receipt",
        {
            "schema",
            "source",
            "invalid_row_policy",
            "rows_read",
            "rows_kept",
            "rows_rejected",
            "kept_row_hashes",
            "rejected_rows",
            "kept_corpus_sha256",
        },
        {"source_locator"},
    )
    if ingestion["schema"] != "kiln.sft-ingestion.v1":
        raise ContractError("unsupported SFT ingestion receipt")
    for field in (
        "invalid_row_policy",
        "rows_read",
        "rows_kept",
        "rows_rejected",
        "kept_corpus_sha256",
    ):
        if ingestion[field] != selection[field]:
            raise ContractError(f"SFT selection field {field} differs from its receipt")
    if ingestion["source"] != data["source_name"]:
        raise ContractError("SFT data source differs from its ingestion receipt")
    if (
        len(ingestion["kept_row_hashes"]) != ingestion["rows_kept"]
        or len(ingestion["rejected_rows"]) != ingestion["rows_rejected"]
    ):
        raise ContractError("SFT ingestion receipt row evidence has the wrong length")
    return data


def _expect_integer(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = 2**64 - 1,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ContractError(f"{label} must be an integer in {minimum}..={maximum}")
    return value


def _expect_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{label} must be a finite number")
    return number


def _require_f32_representable(value: float, label: str) -> float:
    try:
        rounded = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as exc:
        raise ContractError(f"{label} is outside the finite f32 range") from exc
    if not math.isfinite(rounded):
        raise ContractError(f"{label} is outside the finite f32 range")
    return value


def _validate_identity_text(value: Any, label: str, maximum_bytes: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value.encode("utf-8")) > maximum_bytes
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ContractError(
            f"{label} must be non-empty, trimmed, control-free, and at most "
            f"{maximum_bytes} bytes"
        )
    return value


def _canonical_json_value(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if value < -(2**63) or value > 2**64 - 1:
            raise ContractError(f"{label} integer is outside serde_json's admitted range")
        return value
    if isinstance(value, float):
        return _expect_finite_number(value, label)
    if isinstance(value, list):
        return [
            _canonical_json_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ContractError(f"{label} object keys must be strings")
        return {
            key: _canonical_json_value(value[key], f"{label}.{key}")
            for key in sorted(value)
        }
    raise ContractError(f"{label} contains unsupported {type(value).__name__}")


def _canonical_chat_messages(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{label} must contain at least one message")
    messages: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        message_label = f"{label}[{index}]"
        message = _expect_keys(
            raw,
            message_label,
            {"role", "content"},
            {"tool_calls", "name", "tool_call_id"},
        )
        role = _validate_identity_text(message["role"], f"{message_label}.role")
        content = message["content"]
        if not isinstance(content, str):
            raise ContractError(f"{message_label}.content must be a string")
        canonical: dict[str, Any] = {"role": role, "content": content}
        if "tool_calls" in message:
            tool_calls = message["tool_calls"]
            if not isinstance(tool_calls, list):
                raise ContractError(f"{message_label}.tool_calls must be an array")
            canonical["tool_calls"] = [
                _canonical_json_value(call, f"{message_label}.tool_calls[{call_index}]")
                for call_index, call in enumerate(tool_calls)
            ]
        for field in ("name", "tool_call_id"):
            if field in message:
                canonical[field] = _validate_identity_text(
                    message[field], f"{message_label}.{field}"
                )
        messages.append(canonical)
    return messages


def _canonical_trajectory(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be an array")
    trajectory: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        segment_label = f"{label}[{index}]"
        segment = _expect_keys(
            raw,
            segment_label,
            {"role", "content"},
            {"kind", "tool_call_id", "warning_prefix_len"},
        )
        role = _validate_identity_text(segment["role"], f"{segment_label}.role")
        content = segment["content"]
        if not isinstance(content, str):
            raise ContractError(f"{segment_label}.content must be a string")
        kind = segment.get("kind", "context")
        if kind not in ("context", "action", "observation"):
            raise ContractError(f"{segment_label}.kind is unsupported")
        canonical: dict[str, Any] = {
            "role": role,
            "content": content,
            "kind": kind,
        }
        if "tool_call_id" in segment:
            canonical["tool_call_id"] = _validate_identity_text(
                segment["tool_call_id"], f"{segment_label}.tool_call_id"
            )
        if "warning_prefix_len" in segment:
            warning = _expect_integer(
                segment["warning_prefix_len"],
                f"{segment_label}.warning_prefix_len",
            )
            if warning > len(content.encode("utf-8")):
                raise ContractError(
                    f"{segment_label}.warning_prefix_len exceeds the content length"
                )
            canonical["warning_prefix_len"] = warning
        trajectory.append(canonical)
    return trajectory


def _rollout_prompt_sha256(messages: list[dict[str, Any]]) -> str:
    return _sha256_bytes(_compact_json_bytes(messages))


def _scored_payload_sha256(text: str, trajectory: list[dict[str, Any]]) -> str:
    return _sha256_bytes(
        _compact_json_bytes(
            {
                "schema": "kiln.scored-rollout-payload.v1",
                "text": text,
                "trajectory": trajectory,
            }
        )
    )


def _ordered_grpo_corpus_sha256(rows: Any) -> str:
    digest = hashlib.sha256()
    digest.update(GRPO_CORPUS_DOMAIN)
    row_count = 0
    for row_count, row in enumerate(rows, 1):
        digest.update((row_count - 1).to_bytes(8, "little"))
        digest.update(len(row).to_bytes(8, "little"))
        digest.update(row)
    digest.update(row_count.to_bytes(8, "little"))
    return SHA256_PREFIX + digest.hexdigest()


def _expected_behavior_adapter(
    input_adapter: dict[str, Any] | None,
) -> dict[str, str] | None:
    if input_adapter is None:
        return None
    model_identity = input_adapter["model"]
    config_identity = input_adapter["config"]
    raw_weights = bytes.fromhex(
        _validate_sha256(model_identity["sha256"], "input adapter weights hash")[
            len(SHA256_PREFIX) :
        ]
    )
    filename = ADAPTER_MODEL.encode("utf-8")
    weights_digest = hashlib.sha256()
    weights_digest.update(b"kiln.adapter-weights.v1\0")
    weights_digest.update((1).to_bytes(8, "little"))
    weights_digest.update(len(filename).to_bytes(8, "little"))
    weights_digest.update(filename)
    weights_digest.update(model_identity["size_bytes"].to_bytes(8, "little"))
    weights_digest.update(raw_weights)
    weights_identity = weights_digest.hexdigest().encode("ascii")
    config_identity_hex = _validate_sha256(
        config_identity["sha256"], "input adapter config hash"
    )[len(SHA256_PREFIX) :].encode("ascii")
    content_digest = hashlib.sha256()
    content_digest.update(b"kiln.adapter-content-revision.v1\0")
    for identity in (weights_identity, config_identity_hex):
        content_digest.update(len(identity).to_bytes(8, "little"))
        content_digest.update(identity)
    return {
        "name": input_adapter["name"],
        "content_sha256": SHA256_PREFIX + content_digest.hexdigest(),
    }


def _canonical_template_invocation(value: Any, label: str) -> dict[str, Any]:
    invocation = _expect_keys(
        value,
        label,
        set(),
        {"tools", "tool_choice", "template_kwargs"},
    )
    canonical: dict[str, Any] = {}
    if "tools" in invocation:
        tools = invocation["tools"]
        if not isinstance(tools, list) or len(tools) > MAX_GRPO_TEMPLATE_TOOLS:
            raise ContractError(
                f"{label}.tools must be an array with at most {MAX_GRPO_TEMPLATE_TOOLS} entries"
            )
        if tools:
            canonical["tools"] = [
                _canonical_json_value(tool, f"{label}.tools[{index}]")
                for index, tool in enumerate(tools)
            ]
    if "tool_choice" in invocation and invocation["tool_choice"] is not None:
        canonical["tool_choice"] = _canonical_json_value(
            invocation["tool_choice"], f"{label}.tool_choice"
        )
    if "template_kwargs" in invocation:
        kwargs = invocation["template_kwargs"]
        if not isinstance(kwargs, dict) or len(kwargs) > MAX_GRPO_TEMPLATE_KWARGS:
            raise ContractError(
                f"{label}.template_kwargs must be an object with at most "
                f"{MAX_GRPO_TEMPLATE_KWARGS} entries"
            )
        for key in kwargs:
            if not key or len(key.encode("utf-8")) > 256:
                raise ContractError(f"{label}.template_kwargs has an invalid key")
        if kwargs:
            canonical["template_kwargs"] = {
                key: _canonical_json_value(kwargs[key], f"{label}.template_kwargs.{key}")
                for key in sorted(kwargs)
            }
    if len(_compact_json_bytes(canonical)) > MAX_GRPO_TEMPLATE_BYTES:
        raise ContractError(f"{label} exceeds {MAX_GRPO_TEMPLATE_BYTES} serialized bytes")
    return canonical


def _canonical_sampling(value: Any, label: str) -> dict[str, Any]:
    sampling = _expect_keys(
        value,
        label,
        {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "stop",
        },
        {"thinking_budget"},
    )
    temperature = _expect_finite_number(sampling["temperature"], f"{label}.temperature")
    top_p = _expect_finite_number(sampling["top_p"], f"{label}.top_p")
    min_p = _expect_finite_number(sampling["min_p"], f"{label}.min_p")
    repetition = _expect_finite_number(
        sampling["repetition_penalty"], f"{label}.repetition_penalty"
    )
    presence = _expect_finite_number(
        sampling["presence_penalty"], f"{label}.presence_penalty"
    )
    frequency = _expect_finite_number(
        sampling["frequency_penalty"], f"{label}.frequency_penalty"
    )
    if temperature < 0.0:
        raise ContractError(f"{label}.temperature must be non-negative")
    if not 0.0 <= top_p <= 1.0 or not 0.0 <= min_p <= 1.0:
        raise ContractError(f"{label}.top_p and min_p must be within [0, 1]")
    if repetition <= 0.0:
        raise ContractError(f"{label}.repetition_penalty must be positive")
    if not -2.0 <= presence <= 2.0 or not -2.0 <= frequency <= 2.0:
        raise ContractError(f"{label} presence/frequency penalties must be within [-2, 2]")
    stop = sampling["stop"]
    if (
        not isinstance(stop, list)
        or len(stop) > MAX_GRPO_STOP_SEQUENCES
        or any(not isinstance(item, str) or not item for item in stop)
        or sum(len(item.encode("utf-8")) for item in stop) > MAX_GRPO_STOP_BYTES
    ):
        raise ContractError(f"{label}.stop violates the bounded non-empty string contract")
    canonical: dict[str, Any] = {
        "temperature": _F32(temperature),
        "top_p": _F32(top_p),
        "top_k": _expect_integer(
            sampling["top_k"], f"{label}.top_k", maximum=2**32 - 1
        ),
        "min_p": _F32(min_p),
        "max_tokens": _expect_integer(
            sampling["max_tokens"], f"{label}.max_tokens", minimum=1
        ),
        "repetition_penalty": _F32(repetition),
        "presence_penalty": _F32(presence),
        "frequency_penalty": _F32(frequency),
        "stop": list(stop),
    }
    if "thinking_budget" in sampling and sampling["thinking_budget"] is not None:
        budget = _expect_keys(
            sampling["thinking_budget"],
            f"{label}.thinking_budget",
            {"close_token_ids"},
            {"max_tokens", "max_time_ms"},
        )
        canonical_budget: dict[str, Any] = {}
        if "max_tokens" in budget and budget["max_tokens"] is not None:
            canonical_budget["max_tokens"] = _expect_integer(
                budget["max_tokens"], f"{label}.thinking_budget.max_tokens"
            )
        if "max_time_ms" in budget and budget["max_time_ms"] is not None:
            canonical_budget["max_time_ms"] = _expect_integer(
                budget["max_time_ms"], f"{label}.thinking_budget.max_time_ms"
            )
        close_ids = budget["close_token_ids"]
        if not isinstance(close_ids, list) or not close_ids:
            raise ContractError(f"{label}.thinking_budget.close_token_ids must be non-empty")
        canonical_budget["close_token_ids"] = [
            _expect_integer(
                token,
                f"{label}.thinking_budget.close_token_ids[{index}]",
                maximum=2**32 - 1,
            )
            for index, token in enumerate(close_ids)
        ]
        if not any(field in canonical_budget for field in ("max_tokens", "max_time_ms")):
            raise ContractError(f"{label}.thinking_budget must contain a token or time limit")
        canonical["thinking_budget"] = canonical_budget
    return canonical


def _canonical_behavior_policy(
    value: Any,
    label: str,
    model: dict[str, Any],
    expected_adapter: dict[str, str] | None,
) -> dict[str, Any]:
    behavior = _expect_keys(
        value,
        label,
        {
            "served_model_id",
            "base_model_sha256",
            "inference_config_sha256",
            "implementation",
        },
        {"adapter"},
    )
    canonical: dict[str, Any] = {
        "served_model_id": _validate_identity_text(
            behavior["served_model_id"], f"{label}.served_model_id"
        ),
        "base_model_sha256": _validate_sha256(
            behavior["base_model_sha256"], f"{label}.base_model_sha256"
        ),
    }
    if "adapter" in behavior and behavior["adapter"] is not None:
        adapter = _expect_keys(
            behavior["adapter"], f"{label}.adapter", {"name", "content_sha256"}
        )
        canonical["adapter"] = {
            "name": _validate_identity_text(adapter["name"], f"{label}.adapter.name"),
            "content_sha256": _validate_sha256(
                adapter["content_sha256"], f"{label}.adapter.content_sha256"
            ),
        }
    canonical["inference_config_sha256"] = _validate_sha256(
        behavior["inference_config_sha256"], f"{label}.inference_config_sha256"
    )
    canonical["implementation"] = _validate_identity_text(
        behavior["implementation"], f"{label}.implementation"
    )
    if canonical["served_model_id"] != model["served_model_id"]:
        raise ContractError(f"{label} served model differs from the export")
    if (
        canonical["base_model_sha256"]
        != model["base_weight_shard_manifest"]["aggregate_sha256"]
    ):
        raise ContractError(f"{label} base model differs from the export")
    if canonical.get("adapter") != expected_adapter:
        raise ContractError(f"{label} adapter differs from the exported input adapter")
    return canonical


def _canonical_provenance(
    value: Any,
    label: str,
    model: dict[str, Any],
    expected_adapter: dict[str, str] | None,
    prompt_sha256: str,
    payload_sha256: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    provenance = _expect_keys(
        value,
        label,
        {
            "schema",
            "input_token_ids",
            "prompt_token_count",
            "prompt_messages_sha256",
            "scored_payload_sha256",
            "action_tokens",
            "behavior_policy",
            "tokenizer",
            "sampling",
            "seed",
            "generation_backend",
        },
        {"template_invocation"},
    )
    if provenance["schema"] != GRPO_PROVENANCE_SCHEMA:
        raise ContractError(f"{label} uses an unsupported schema")
    input_ids = provenance["input_token_ids"]
    if not isinstance(input_ids, list) or not 1 <= len(input_ids) <= MAX_GRPO_TOKEN_COUNT:
        raise ContractError(f"{label}.input_token_ids has an invalid length")
    canonical_ids = [
        _expect_integer(token, f"{label}.input_token_ids[{index}]", maximum=2**32 - 1)
        for index, token in enumerate(input_ids)
    ]
    prompt_count = _expect_integer(
        provenance["prompt_token_count"],
        f"{label}.prompt_token_count",
        minimum=1,
        maximum=len(canonical_ids),
    )
    declared_prompt = _validate_sha256(
        provenance["prompt_messages_sha256"], f"{label}.prompt_messages_sha256"
    )
    declared_payload = _validate_sha256(
        provenance["scored_payload_sha256"], f"{label}.scored_payload_sha256"
    )
    if declared_prompt != prompt_sha256:
        raise ContractError(f"{label} prompt identity differs from its group")
    if declared_payload != payload_sha256:
        raise ContractError(f"{label} scored payload identity differs from its completion")
    actions = provenance["action_tokens"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= len(canonical_ids):
        raise ContractError(f"{label}.action_tokens has an invalid length")
    canonical_actions: list[dict[str, Any]] = []
    previous = -1
    sampled = 0
    forced = 0
    for index, raw in enumerate(actions):
        action_label = f"{label}.action_tokens[{index}]"
        action = _expect_keys(
            raw,
            action_label,
            {"sequence_index", "token_id", "source", "behavior_logprob"},
        )
        sequence_index = _expect_integer(
            action["sequence_index"],
            f"{action_label}.sequence_index",
            minimum=prompt_count,
            maximum=len(canonical_ids) - 1,
        )
        token_id = _expect_integer(
            action["token_id"], f"{action_label}.token_id", maximum=2**32 - 1
        )
        if sequence_index <= previous:
            raise ContractError(f"{label} action-token indices are not strictly increasing")
        if canonical_ids[sequence_index] != token_id:
            raise ContractError(f"{action_label} token ID differs from input_token_ids")
        previous = sequence_index
        source = action["source"]
        logprob = action["behavior_logprob"]
        canonical_action: dict[str, Any] = {
            "sequence_index": sequence_index,
            "token_id": token_id,
            "source": source,
        }
        if source == "sampled":
            logprob = _require_f32_representable(
                _expect_finite_number(
                    logprob, f"{action_label}.behavior_logprob"
                ),
                f"{action_label}.behavior_logprob",
            )
            if logprob > 1e-6:
                raise ContractError(f"{action_label}.behavior_logprob must not be positive")
            canonical_action["behavior_logprob"] = logprob
            sampled += 1
        elif source == "forced":
            if logprob is not None:
                raise ContractError(f"{action_label} forced token must have null behavior_logprob")
            canonical_action["behavior_logprob"] = None
            forced += 1
        else:
            raise ContractError(f"{action_label}.source is unsupported")
        canonical_actions.append(canonical_action)
    if sampled == 0:
        raise ContractError(f"{label} contains no sampled action token")
    behavior = _canonical_behavior_policy(
        provenance["behavior_policy"], f"{label}.behavior_policy", model, expected_adapter
    )
    tokenizer = _expect_keys(
        provenance["tokenizer"],
        f"{label}.tokenizer",
        {"vocab_sha256", "config_sha256", "chat_template_sha256"},
    )
    canonical_tokenizer = {
        "vocab_sha256": _validate_sha256(
            tokenizer["vocab_sha256"], f"{label}.tokenizer.vocab_sha256"
        ),
        "config_sha256": _validate_sha256(
            tokenizer["config_sha256"], f"{label}.tokenizer.config_sha256"
        ),
        "chat_template_sha256": _validate_sha256(
            tokenizer["chat_template_sha256"],
            f"{label}.tokenizer.chat_template_sha256",
        ),
    }
    expected_tokenizer = {
        "vocab_sha256": model["tokenizer_vocab_sha256"],
        "config_sha256": model["tokenizer"]["sha256"],
        "chat_template_sha256": model["chat_template"]["sha256"],
    }
    if canonical_tokenizer != expected_tokenizer:
        raise ContractError(f"{label} tokenizer identity differs from the export")
    canonical: dict[str, Any] = {
        "schema": GRPO_PROVENANCE_SCHEMA,
        "input_token_ids": canonical_ids,
        "prompt_token_count": prompt_count,
        "prompt_messages_sha256": declared_prompt,
        "scored_payload_sha256": declared_payload,
        "action_tokens": canonical_actions,
        "behavior_policy": behavior,
        "tokenizer": canonical_tokenizer,
    }
    if "template_invocation" in provenance:
        invocation = _canonical_template_invocation(
            provenance["template_invocation"], f"{label}.template_invocation"
        )
        if invocation:
            canonical["template_invocation"] = invocation
    sampling = _canonical_sampling(provenance["sampling"], f"{label}.sampling")
    canonical["sampling"] = sampling
    canonical["seed"] = _expect_integer(provenance["seed"], f"{label}.seed")
    canonical["generation_backend"] = _validate_identity_text(
        provenance["generation_backend"], f"{label}.generation_backend", 64
    )
    if forced:
        budget = sampling.get("thinking_budget")
        if budget is None:
            raise ContractError(f"{label} forced tokens require thinking-budget provenance")
        close_ids = set(budget["close_token_ids"])
        if any(
            action["source"] == "forced" and action["token_id"] not in close_ids
            for action in canonical_actions
        ):
            raise ContractError(f"{label} forced token is absent from thinking-budget close IDs")
    return canonical, {
        "sampled": sampled,
        "forced": forced,
        "sequence_tokens": len(canonical_ids),
        "completion_tokens": len(canonical_ids) - prompt_count,
        "unclassified_completion_tokens": (
            len(canonical_ids) - prompt_count - len(canonical_actions)
        ),
    }


def _canonical_grpo_group(
    value: Any,
    row_index: int,
    model: dict[str, Any],
    expected_adapter: dict[str, str] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = f"GRPO row {row_index}"
    group = _expect_keys(value, label, {"messages", "completions"})
    messages = _canonical_chat_messages(group["messages"], f"{label}.messages")
    completions = group["completions"]
    if not isinstance(completions, list) or not 2 <= len(completions) <= MAX_GRPO_COMPLETIONS:
        raise ContractError(f"{label} must contain 2..={MAX_GRPO_COMPLETIONS} completions")
    prompt_sha256 = _rollout_prompt_sha256(messages)
    canonical_completions: list[dict[str, Any]] = []
    behavior_policy: dict[str, Any] | None = None
    sampled = 0
    forced = 0
    max_sequence = 0
    max_completion = 0
    for index, raw in enumerate(completions):
        completion_label = f"{label}.completions[{index}]"
        completion = _expect_keys(
            raw,
            completion_label,
            {"text", "reward", "provenance"},
            {"trajectory"},
        )
        text = completion["text"]
        if not isinstance(text, str):
            raise ContractError(f"{completion_label}.text must be a string")
        reward = _require_f32_representable(
            _expect_finite_number(
                completion["reward"], f"{completion_label}.reward"
            ),
            f"{completion_label}.reward",
        )
        trajectory = _canonical_trajectory(
            completion.get("trajectory", []), f"{completion_label}.trajectory"
        )
        provenance, counts = _canonical_provenance(
            completion["provenance"],
            f"{completion_label}.provenance",
            model,
            expected_adapter,
            prompt_sha256,
            _scored_payload_sha256(text, trajectory),
        )
        if counts["unclassified_completion_tokens"] and not any(
            segment["kind"] == "observation" for segment in trajectory
        ):
            raise ContractError(
                f"{completion_label} leaves suffix tokens outside action provenance "
                "without a trajectory observation"
            )
        current_behavior = provenance["behavior_policy"]
        if behavior_policy is None:
            behavior_policy = current_behavior
        elif behavior_policy != current_behavior:
            raise ContractError(f"{label} mixes behavior-policy identities")
        canonical_completion: dict[str, Any] = {"text": text, "reward": reward}
        if trajectory:
            canonical_completion["trajectory"] = trajectory
        canonical_completion["provenance"] = provenance
        canonical_completions.append(canonical_completion)
        sampled += counts["sampled"]
        forced += counts["forced"]
        max_sequence = max(max_sequence, counts["sequence_tokens"])
        max_completion = max(max_completion, counts["completion_tokens"])
    return {
        "messages": messages,
        "completions": canonical_completions,
    }, {
        "completion_count": len(completions),
        "sampled_action_tokens": sampled,
        "forced_action_tokens": forced,
        "max_sequence_tokens": max_sequence,
        "max_completion_tokens": max_completion,
        "behavior_policy": behavior_policy,
    }


def _iter_grpo_rows(path: Path):
    try:
        with path.open("rb") as handle:
            row_index = 0
            while True:
                raw = handle.readline(MAX_GRPO_ROW_BYTES + 1)
                if not raw:
                    return
                row_index += 1
                if len(raw) > MAX_GRPO_ROW_BYTES:
                    raise ContractError(
                        f"GRPO row {row_index} exceeds {MAX_GRPO_ROW_BYTES} bytes"
                    )
                if not raw.endswith(b"\n"):
                    raise ContractError(
                        "GRPO dataset must end every row with LF, including the final row"
                    )
                canonical_bytes = raw[:-1]
                if not canonical_bytes:
                    raise ContractError(f"GRPO dataset contains a blank row at position {row_index}")
                try:
                    value = json.loads(
                        canonical_bytes.decode("utf-8"), object_pairs_hook=_unique_object
                    )
                except ContractError:
                    raise
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise ContractError(f"cannot parse GRPO row {row_index}: {exc}") from exc
                yield row_index, canonical_bytes, value
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError(f"cannot read GRPO dataset {path}: {exc}") from exc


def _scan_grpo_dataset(
    path: Path,
    model: dict[str, Any],
    input_adapter: dict[str, Any] | None,
) -> dict[str, Any]:
    expected_adapter = _expected_behavior_adapter(input_adapter)
    digest = hashlib.sha256()
    digest.update(GRPO_CORPUS_DOMAIN)
    row_count = 0
    completion_count = 0
    sampled = 0
    forced = 0
    max_sequence = 0
    max_completion = 0
    num_generations: int | None = None
    behavior_policy: dict[str, Any] | None = None
    for row_index, raw, parsed in _iter_grpo_rows(path):
        if row_index > MAX_GRPO_GROUPS:
            raise ContractError(f"GRPO dataset exceeds {MAX_GRPO_GROUPS} groups")
        canonical, summary = _canonical_grpo_group(
            parsed, row_index, model, expected_adapter
        )
        encoded = _compact_json_bytes(canonical)
        if encoded != raw:
            raise ContractError(
                f"GRPO row {row_index} is not canonical compact JSON or contains "
                "unknown, duplicate, default-valued, or misordered fields"
            )
        digest.update((row_index - 1).to_bytes(8, "little"))
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
        row_count = row_index
        completion_count += summary["completion_count"]
        sampled += summary["sampled_action_tokens"]
        forced += summary["forced_action_tokens"]
        max_sequence = max(max_sequence, summary["max_sequence_tokens"])
        max_completion = max(max_completion, summary["max_completion_tokens"])
        if num_generations is None:
            num_generations = summary["completion_count"]
        elif num_generations != summary["completion_count"]:
            raise ContractError(
                "the pinned TRL 1.8.0 reference runner requires one uniform completion "
                "count across all GRPO groups"
            )
        if behavior_policy is None:
            behavior_policy = summary["behavior_policy"]
        elif behavior_policy != summary["behavior_policy"]:
            raise ContractError(
                f"GRPO row {row_index} uses a different behavior-policy identity"
            )
    if row_count == 0:
        raise ContractError("GRPO dataset contains no groups")
    digest.update(row_count.to_bytes(8, "little"))
    return {
        "row_count": row_count,
        "completion_count": completion_count,
        "sampled_action_tokens": sampled,
        "forced_action_tokens": forced,
        "max_sequence_tokens": max_sequence,
        "max_completion_tokens": max_completion,
        "num_generations": num_generations,
        "ordered_corpus_sha256": SHA256_PREFIX + digest.hexdigest(),
        "behavior_policy": behavior_policy,
    }


def _validate_grpo_export_data(
    root: Path,
    value: Any,
    model: dict[str, Any],
    input_adapter: Any,
) -> dict[str, Any]:
    data = _expect_keys(
        value,
        "export data",
        {
            "source_name",
            "format",
            "row_count",
            "ordered_corpus_sha256",
            "dataset",
            "rollout_provenance_schema",
        },
        {"split_manifest"},
    )
    if (
        data["format"] != "grpo_groups_jsonl"
        or data["rollout_provenance_schema"] != GRPO_PROVENANCE_SCHEMA
        or not isinstance(data["row_count"], int)
        or isinstance(data["row_count"], bool)
        or not 1 <= data["row_count"] <= MAX_GRPO_GROUPS
    ):
        raise ContractError("invalid GRPO data format, provenance schema, or row_count")
    _validate_sha256(data["ordered_corpus_sha256"], "GRPO ordered corpus digest")
    dataset_identity = _expect_object(data["dataset"], "GRPO dataset identity")
    size_bytes = dataset_identity.get("size_bytes")
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or not 1 <= size_bytes <= MAX_GRPO_DATASET_BYTES
    ):
        raise ContractError(
            f"GRPO dataset size must be within 1..={MAX_GRPO_DATASET_BYTES} bytes"
        )
    dataset_path = _verify_identity(root, data["dataset"], "train.jsonl")
    summary = _scan_grpo_dataset(dataset_path, model, input_adapter)
    if summary["row_count"] != data["row_count"]:
        raise ContractError(
            f"GRPO row count {summary['row_count']} differs from manifest value {data['row_count']}"
        )
    if summary["ordered_corpus_sha256"] != data["ordered_corpus_sha256"]:
        raise ContractError("GRPO ordered corpus identity differs from its manifest")
    return data


def load_export_bundle(bundle: Path) -> tuple[Path, dict[str, Any]]:
    root = _require_real_bundle_root(bundle)
    manifest_path = _bundle_file(root, EXPORT_MANIFEST)
    manifest = _expect_keys(
        _read_json(manifest_path, bounded=True),
        "HF/TRL export",
        {
            "schema_version",
            "manifest_type",
            "task",
            "source_execution_provenance",
            "model",
            "data",
            "reference_script",
            "environment_lock",
            "export_sha256",
        },
        {"input_adapter"},
    )
    if manifest["schema_version"] != 1 or manifest["manifest_type"] != "kiln.hf-trl-export.v1":
        raise ContractError("unsupported HF/TRL export schema")
    if manifest["task"] not in ("sft", "grpo"):
        raise ContractError(f"unsupported HF/TRL task {manifest['task']!r}")
    expected_digest = _validate_sha256(manifest["export_sha256"], "export_sha256")
    digest_fields = dict(manifest)
    del digest_fields["export_sha256"]
    actual_digest = _canonical_sha256(digest_fields)
    if actual_digest != expected_digest:
        raise ContractError(
            f"export manifest digest differs: manifest={expected_digest}, expected={actual_digest}"
        )

    provenance = _validate_source_provenance(manifest["source_execution_provenance"])
    model = _expect_keys(
        manifest["model"],
        "export model",
        {
            "served_model_id",
            "base_weight_shard_manifest",
            "tokenizer_vocab_sha256",
            "model_config",
            "tokenizer",
            "chat_template",
            "native_training_chat_template",
            "trl_training_chat_template",
        },
    )
    base = _expect_keys(
        model["base_weight_shard_manifest"],
        "base-weight manifest",
        {
            "schema_version",
            "manifest_type",
            "aggregate_algorithm",
            "aggregate_sha256",
            "total_size_bytes",
            "shards",
        },
    )
    if (
        base["schema_version"] != 1
        or base["manifest_type"] != "kiln.base-weight-shards.v1"
        or base["aggregate_algorithm"] != "kiln.base-model-content.v1"
    ):
        raise ContractError("unsupported base-weight manifest")
    if (
        not isinstance(base["shards"], list)
        or not 1 <= len(base["shards"]) <= 4096
    ):
        raise ContractError("base-weight manifest must contain 1..=4096 shards")
    prior = ""
    total = 0
    for index, raw_shard in enumerate(base["shards"]):
        shard = _expect_keys(raw_shard, f"base shard {index}", {"filename", "size_bytes", "sha256"})
        filename = shard["filename"]
        if (
            not isinstance(filename, str)
            or PurePosixPath(filename).name != filename
            or not 1 <= len(filename.encode("utf-8")) <= 255
            or not filename.endswith(".safetensors")
            or any(
                not (
                    character.isascii()
                    and (character.isalnum() or character in ".-_")
                )
                for character in filename
            )
            or filename <= prior
        ):
            raise ContractError(f"invalid or unsorted base shard filename {filename!r}")
        size = shard["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ContractError(f"base shard {filename!r} has invalid size")
        _validate_sha256(shard["sha256"], f"base shard {filename!r} hash")
        total += size
        if total > 2**64 - 1:
            raise ContractError("base-weight total_size_bytes exceeds u64")
        prior = filename
    if base["total_size_bytes"] != total:
        raise ContractError("base-weight total_size_bytes differs from its shards")
    aggregate = _validate_sha256(
        base["aggregate_sha256"], "base-weight aggregate digest"
    )
    expected_aggregate = _base_weight_aggregate_sha256(base["shards"])
    if aggregate != expected_aggregate:
        raise ContractError(
            f"base-weight aggregate differs: manifest={aggregate}, "
            f"expected={expected_aggregate}"
        )

    model_paths = {
        "kiln_model_config.json": model["model_config"],
        "tokenizer.json": model["tokenizer"],
        "chat_template.jinja": model["chat_template"],
        "kiln_training_chat_template.jinja": model["native_training_chat_template"],
        "training_chat_template.jinja": model["trl_training_chat_template"],
    }
    for expected_path, identity in model_paths.items():
        _verify_identity(root, identity, expected_path)
    source_model = provenance["model"]
    cross_checks = {
        "model_config_sha256": model["model_config"]["sha256"],
        "tokenizer_config_sha256": model["tokenizer"]["sha256"],
        "tokenizer_vocab_sha256": model["tokenizer_vocab_sha256"],
        "chat_template_sha256": model["chat_template"]["sha256"],
        "training_chat_template_sha256": model["native_training_chat_template"]["sha256"],
    }
    for field, expected in cross_checks.items():
        if source_model.get(field) != expected:
            raise ContractError(f"source provenance {field} differs from exported model")

    adapter = _validate_input_adapter(root, manifest.get("input_adapter"))
    if manifest["task"] == "sft":
        data = _validate_sft_export_data(root, manifest["data"])
    else:
        data = _validate_grpo_export_data(root, manifest["data"], model, adapter)
    if "split_manifest" in data:
        _verify_identity(root, data["split_manifest"], "split_manifest.json")
    _verify_identity(root, manifest["reference_script"], "train.py")
    _verify_identity(root, manifest["environment_lock"], "requirements.lock")

    _declared_export_files(root, manifest)
    if manifest["task"] == "sft":
        rows = _load_dataset_rows(_bundle_file(root, "train.jsonl"))
        if len(rows) != data["row_count"]:
            raise ContractError(
                f"train.jsonl has {len(rows)} rows; manifest declares {data['row_count']}"
            )
        template = _bundle_file(root, "training_chat_template.jinja").read_text(
            encoding="utf-8"
        )
        if "{% generation %}" not in template and "{%- generation %}" not in template:
            raise ContractError("training template has no generation block")
        if "{% endgeneration %}" not in template and "{%- endgeneration %}" not in template:
            raise ContractError("training template has no endgeneration block")
    _validate_environment_lock(_bundle_file(root, "requirements.lock"))
    return root, manifest


def _load_dataset_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    raise ContractError(f"train.jsonl line {line_number} is blank")
                row = json.loads(raw, object_pairs_hook=_unique_object)
                row = _expect_keys(row, f"train.jsonl line {line_number}", {"messages"})
                messages = row["messages"]
                if not isinstance(messages, list) or not messages:
                    raise ContractError(f"train.jsonl line {line_number} has no messages")
                rows.append(row)
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot parse {path}: {exc}") from exc
    return rows


def _assert_grpo_dataset_identity(path: Path, identity: dict[str, Any]) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ContractError(f"cannot stat recorded GRPO dataset {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"recorded GRPO dataset is not a real regular file: {path}")
    if metadata.st_size != identity["size_bytes"]:
        raise ContractError("recorded GRPO dataset size changed after verification")
    if _sha256_file(path) != identity["sha256"]:
        raise ContractError("recorded GRPO dataset hash changed after verification")


def _grpo_prompt_records(
    path: Path,
    model: dict[str, Any],
    input_adapter: dict[str, Any] | None,
):
    expected_adapter = _expected_behavior_adapter(input_adapter)
    for row_index, raw, parsed in _iter_grpo_rows(path):
        canonical, _ = _canonical_grpo_group(
            parsed, row_index, model, expected_adapter
        )
        if _compact_json_bytes(canonical) != raw:
            raise ContractError(f"GRPO row {row_index} changed after verification")
        yield {"prompt": _compact_json_bytes(canonical["messages"]).decode("utf-8")}


class _RecordedRolloutSource:
    """Replay exact token decisions from one verified immutable GRPO snapshot."""

    def __init__(
        self,
        path: Path,
        dataset_identity: dict[str, Any],
        model: dict[str, Any],
        input_adapter: dict[str, Any] | None,
        num_generations: int,
    ) -> None:
        self.path = path
        self.dataset_identity = dataset_identity
        self.model = model
        self.expected_adapter = _expected_behavior_adapter(input_adapter)
        self.num_generations = num_generations
        self.groups_served = 0
        self.epochs_opened = 0
        self._rows = None

    def _open_epoch(self) -> None:
        _assert_grpo_dataset_identity(self.path, self.dataset_identity)
        self._rows = iter(_iter_grpo_rows(self.path))
        self.epochs_opened += 1

    def _next_group(self) -> dict[str, Any]:
        if self._rows is None:
            self._open_epoch()
        try:
            row_index, raw, parsed = next(self._rows)
        except StopIteration:
            self._open_epoch()
            try:
                row_index, raw, parsed = next(self._rows)
            except StopIteration as exc:
                raise ContractError("recorded GRPO dataset became empty") from exc
        canonical, summary = _canonical_grpo_group(
            parsed, row_index, self.model, self.expected_adapter
        )
        if _compact_json_bytes(canonical) != raw:
            raise ContractError(f"GRPO row {row_index} changed after verification")
        if summary["completion_count"] != self.num_generations:
            raise ContractError(
                f"GRPO row {row_index} completion count changed after verification"
            )
        self.groups_served += 1
        return canonical

    def __call__(self, prompts: list[Any], trainer: Any) -> dict[str, Any]:
        del trainer
        if not prompts or len(prompts) % self.num_generations != 0:
            raise ContractError(
                "TRL supplied a recorded-rollout batch that does not contain complete groups"
            )
        prompt_ids: list[list[int]] = []
        completion_ids: list[list[int]] = []
        logprobs: list[list[float]] = []
        env_mask: list[list[int]] = []
        recorded_reward: list[float] = []
        for offset in range(0, len(prompts), self.num_generations):
            group = self._next_group()
            expected_prompt = _compact_json_bytes(group["messages"]).decode("utf-8")
            supplied = prompts[offset : offset + self.num_generations]
            if any(prompt != expected_prompt for prompt in supplied):
                raise ContractError(
                    "TRL prompt order or repetition differs from the verified GRPO corpus"
                )
            for completion in group["completions"]:
                provenance = completion["provenance"]
                boundary = provenance["prompt_token_count"]
                full_ids = provenance["input_token_ids"]
                suffix = full_ids[boundary:]
                suffix_logprobs = [0.0] * len(suffix)
                suffix_mask = [0] * len(suffix)
                for action in provenance["action_tokens"]:
                    position = action["sequence_index"] - boundary
                    if action["source"] == "sampled":
                        suffix_logprobs[position] = action["behavior_logprob"]
                        suffix_mask[position] = 1
                prompt_ids.append(list(full_ids[:boundary]))
                completion_ids.append(list(suffix))
                logprobs.append(suffix_logprobs)
                env_mask.append(suffix_mask)
                recorded_reward.append(completion["reward"])
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "env_mask": env_mask,
            "recorded_reward": recorded_reward,
        }


def _recorded_reward(*, recorded_reward: list[float], **_: Any) -> list[float]:
    return list(recorded_reward)


def _validate_environment_lock(path: Path) -> None:
    pins: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read environment lock {path}: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line or any(mark in line for mark in (";", "[", "]", "@", " ")):
            raise ContractError(f"unsupported requirements.lock line {line_number}: {raw!r}")
        name, version = line.split("==", 1)
        normalized = name.lower().replace("_", "-")
        if not name or not version or normalized in pins:
            raise ContractError(f"invalid or duplicate requirements.lock line {line_number}")
        pins[normalized] = version
    if pins != PINNED_PACKAGES:
        raise ContractError(f"requirements.lock pins {pins!r}; expected {PINNED_PACKAGES!r}")


def verify_base_model_source(base_model: Path, manifest: dict[str, Any]) -> Path:
    try:
        root = base_model.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"cannot resolve base model {base_model}: {exc}") from exc
    if not root.is_dir():
        raise ContractError(f"base model must be a local directory: {root}")
    base = manifest["model"]["base_weight_shard_manifest"]
    for shard in base["shards"]:
        path = root / shard["filename"]
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ContractError(f"cannot resolve base shard {path}: {exc}") from exc
        if not resolved.is_file() or resolved.stat().st_size != shard["size_bytes"]:
            raise ContractError(f"base shard {path} is missing or has the wrong size")
        actual = _sha256_file(resolved)
        if actual != shard["sha256"]:
            raise ContractError(
                f"base shard {shard['filename']!r} differs: manifest={shard['sha256']}, actual={actual}"
            )
    tokenizer_path = root / "tokenizer.json"
    try:
        tokenizer_path = tokenizer_path.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"cannot resolve base tokenizer {tokenizer_path}: {exc}") from exc
    tokenizer_hash = _sha256_file(tokenizer_path)
    expected_tokenizer_hash = manifest["model"]["tokenizer"]["sha256"]
    if tokenizer_hash != expected_tokenizer_hash:
        raise ContractError(
            f"base tokenizer differs: manifest={expected_tokenizer_hash}, actual={tokenizer_hash}"
        )
    config = root / "config.json"
    if not config.is_file():
        raise ContractError(f"base model has no regular config.json: {root}")
    return root


def _normalize_messages_for_hf(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        if not isinstance(message, dict):
            raise ContractError("every SFT message must be an object")
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue
        if message.get("content") == "":
            message["content"] = None
        for tool_call in tool_calls:
            function = tool_call.get("function", tool_call)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments, object_pairs_hook=_unique_object)
                except json.JSONDecodeError as exc:
                    raise ContractError(f"tool arguments are not valid JSON: {arguments!r}") from exc
                if not isinstance(arguments, dict):
                    raise ContractError("tool-call arguments must decode to an object")
                function["arguments"] = arguments
    return normalized


def _decimal(value: str, label: str, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ContractError(f"{label} is not a decimal number: {value!r}") from exc
    if not math.isfinite(parsed) or parsed < minimum or (maximum is not None and parsed > maximum):
        raise ContractError(f"{label} must be finite and in range [{minimum}, {maximum}]")
    return parsed


def _assert_installed_pins() -> None:
    actual: dict[str, str] = {}
    for package in PINNED_PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ContractError(f"pinned package {package} is not installed") from exc
        actual[package] = version.split("+", 1)[0]
    if actual != PINNED_PACKAGES:
        raise ContractError(f"installed training packages are {actual!r}; expected {PINNED_PACKAGES!r}")


def _file_identity(path: Path, relative: str) -> dict[str, Any]:
    size = path.stat().st_size
    if size <= 0:
        raise ContractError(f"result artifact {relative} is empty")
    return {"relative_path": relative, "size_bytes": size, "sha256": _sha256_file(path)}


def _config_value(kind: str, value: Any) -> dict[str, Any]:
    return {"kind": kind, "value": value}


def _validate_bounded_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value.encode("utf-8")) > 512
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)
    ):
        raise ContractError(f"{label} must be non-empty, trimmed, bounded text")
    return value


def _validate_effective_config(config: dict[str, dict[str, Any]]) -> None:
    if not config or len(config) > 256:
        raise ContractError("effective trainer config must contain 1..=256 entries")
    for key, raw in config.items():
        _validate_bounded_text(key, "effective config key")
        value = _expect_keys(raw, f"effective config {key!r}", {"kind", "value"})
        kind = value["kind"]
        item = value["value"]
        if kind == "boolean":
            valid = isinstance(item, bool)
        elif kind == "integer":
            valid = isinstance(item, int) and not isinstance(item, bool) and -(2**63) <= item < 2**63
        elif kind == "unsigned":
            valid = isinstance(item, int) and not isinstance(item, bool) and 0 <= item < 2**64
        elif kind == "decimal":
            valid = isinstance(item, str)
            if valid:
                _validate_bounded_text(item, f"effective config {key!r} decimal")
                try:
                    valid = math.isfinite(float(item))
                except ValueError:
                    valid = False
        elif kind == "text":
            _validate_bounded_text(item, f"effective config {key!r} text")
            valid = True
        elif kind == "text_list":
            valid = isinstance(item, list) and len(item) <= 256
            if valid:
                for index, text in enumerate(item):
                    _validate_bounded_text(text, f"effective config {key!r}[{index}]")
        else:
            valid = False
        if not valid:
            raise ContractError(f"effective config {key!r} has invalid {kind!r} value")
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > 64 * 1024:
        raise ContractError("effective trainer config exceeds 65536 bytes")


def _write_new_synced(path: Path, data: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ContractError(f"cannot create result artifact {path}: {exc}") from exc


def _copy_new_synced(source: Path, target: Path) -> None:
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
    except OSError as exc:
        raise ContractError(f"cannot publish result artifact {source} -> {target}: {exc}") from exc


def _recover_or_reject_result_state(root: Path) -> None:
    result = root / RESULT_MANIFEST
    sentinel = root / RESULT_SENTINEL
    outputs = [root / EXECUTED_SCRIPT, root / ADAPTER_CONFIG, root / ADAPTER_MODEL]
    if result.exists():
        raise ContractError(f"bundle already contains {RESULT_MANIFEST}; use a fresh export copy")
    if sentinel.exists():
        if not sentinel.is_file() or sentinel.is_symlink():
            raise ContractError(f"invalid result sentinel {sentinel}")
        for path in outputs:
            if path.exists() and (not path.is_file() or path.is_symlink()):
                raise ContractError(f"cannot recover non-regular partial result {path}")
            path.unlink(missing_ok=True)
        sentinel.unlink()
    elif any(path.exists() for path in outputs):
        raise ContractError("bundle has unattributed result artifacts without an incomplete sentinel")


def _publish_result(
    root: Path,
    manifest: dict[str, Any],
    adapter_dir: Path,
    effective_config: dict[str, dict[str, Any]],
) -> None:
    _validate_effective_config(effective_config)
    _recover_or_reject_result_state(root)
    script_source = Path(__file__).resolve(strict=True)
    source_config = adapter_dir / ADAPTER_CONFIG
    source_model = adapter_dir / ADAPTER_MODEL
    if not source_config.is_file() or source_config.is_symlink():
        raise ContractError("PEFT output has no regular adapter_config.json")
    if not source_model.is_file() or source_model.is_symlink():
        raise ContractError("PEFT output has no regular adapter_model.safetensors")

    sentinel = root / RESULT_SENTINEL
    _write_new_synced(sentinel, b"kiln.hf-trl-result.incomplete.v1\n")
    published = [root / EXECUTED_SCRIPT, root / ADAPTER_CONFIG, root / ADAPTER_MODEL]
    try:
        _copy_new_synced(script_source, root / EXECUTED_SCRIPT)
        _copy_new_synced(source_config, root / ADAPTER_CONFIG)
        _copy_new_synced(source_model, root / ADAPTER_MODEL)
        task = manifest["task"]
        trainer = {
            "kind": "trl_sft_trainer" if task == "sft" else "trl_grpo_trainer",
            "python_version": sys.version.split()[0],
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers"),
            "trl_version": importlib.metadata.version("trl"),
            "peft_version": importlib.metadata.version("peft"),
            "script": _file_identity(root / EXECUTED_SCRIPT, EXECUTED_SCRIPT),
        }
        result: dict[str, Any] = {
            "schema_version": 1,
            "result_type": "kiln.hf-trl-result.v1",
            "export_sha256": manifest["export_sha256"],
            "task": task,
            "trainer": trainer,
            "effective_config": effective_config,
            "output_adapter": {
                "config": _file_identity(root / ADAPTER_CONFIG, ADAPTER_CONFIG),
                "model": _file_identity(root / ADAPTER_MODEL, ADAPTER_MODEL),
            },
        }
        result["result_sha256"] = _canonical_sha256(result)
        encoded = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        _write_new_synced(root / RESULT_MANIFEST, encoded)
        sentinel.unlink()
        directory = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        (root / RESULT_MANIFEST).unlink(missing_ok=True)
        for path in published:
            path.unlink(missing_ok=True)
        sentinel.unlink(missing_ok=True)
        raise


def _script_matches_export(root: Path, manifest: dict[str, Any]) -> bool:
    return _sha256_file(Path(__file__).resolve(strict=True)) == manifest["reference_script"]["sha256"]


def _run_grpo_training(
    args: argparse.Namespace,
    root: Path,
    manifest: dict[str, Any],
    base_model: Path,
) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    if args.max_length is not None:
        raise ContractError(
            "--max-length is SFT-only; recorded GRPO uses the exact provenance token boundary"
        )
    learning_rate = _decimal(args.learning_rate, "learning rate", minimum=0.0)
    epochs = _decimal(args.epochs, "epochs", minimum=0.0)
    beta = _decimal(args.beta, "GRPO beta", minimum=0.0)
    epsilon = _decimal(args.epsilon, "GRPO epsilon", minimum=0.0, maximum=1.0)
    epsilon_high_text = args.epsilon_high or args.epsilon
    epsilon_high = _decimal(
        epsilon_high_text,
        "GRPO epsilon high",
        minimum=0.0,
    )
    if learning_rate <= 0.0 or epochs <= 0.0:
        raise ContractError("learning rate and epochs must be greater than zero")
    weight_decay = _decimal(args.weight_decay, "weight decay", minimum=0.0)
    warmup_ratio = _decimal(args.warmup_ratio, "warmup ratio", minimum=0.0, maximum=1.0)
    max_grad_norm = _decimal(args.max_grad_norm, "max grad norm", minimum=0.0)
    input_adapter = manifest.get("input_adapter")
    dataset_identity = manifest["data"]["dataset"]

    work_parent = root.parent
    work_dir = Path(tempfile.mkdtemp(prefix=f".{root.name}.grpo-work-", dir=work_parent))
    try:
        snapshot = work_dir / "train.jsonl"
        _copy_new_synced(root / "train.jsonl", snapshot)
        _assert_grpo_dataset_identity(snapshot, dataset_identity)
        summary = _scan_grpo_dataset(snapshot, manifest["model"], input_adapter)
        num_generations = summary["num_generations"]
        if summary["row_count"] % args.batch_size != 0:
            raise ContractError(
                f"--batch-size={args.batch_size} groups would make pinned TRL 1.8.0 silently "
                f"drop {summary['row_count'] % args.batch_size} of {summary['row_count']} groups; "
                "choose a positive divisor of the corpus row count"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            str(base_model),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        if not getattr(tokenizer, "is_fast", False):
            raise ContractError("pinned GRPO route requires the exported fast tokenizer")
        tokenizer.chat_template = (root / "chat_template.jinja").read_text(encoding="utf-8")
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ContractError("base tokenizer defines neither pad_token nor eos_token")
            tokenizer.pad_token = tokenizer.eos_token

        if args.dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype_name, dtype = "bfloat16", torch.bfloat16
            elif torch.cuda.is_available() or torch.backends.mps.is_available():
                dtype_name, dtype = "float16", torch.float16
            else:
                dtype_name, dtype = "float32", torch.float32
        else:
            dtype_name = args.dtype
            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[dtype_name]
        bf16 = dtype_name == "bfloat16"
        fp16 = dtype_name == "float16"

        set_seed(args.seed)
        model = AutoModelForCausalLM.from_pretrained(
            str(base_model),
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
        )
        model.config.use_cache = not args.gradient_checkpointing

        peft_config = None
        if input_adapter is not None:
            if any(
                value is not None
                for value in (
                    args.lora_rank,
                    args.lora_alpha,
                    args.lora_dropout,
                    args.target_modules,
                )
            ):
                raise ContractError(
                    "LoRA creation options cannot be combined with an exported input adapter"
                )
            model = PeftModel.from_pretrained(
                model, str(root / "input_adapter"), is_trainable=True
            )
            rank = alpha = None
            dropout_text = None
            target_modules = None
        else:
            rank = args.lora_rank or 16
            alpha = args.lora_alpha or (rank * 2)
            dropout_text = args.lora_dropout or "0.0"
            dropout = _decimal(
                dropout_text, "LoRA dropout", minimum=0.0, maximum=1.0
            )
            target_modules = _target_modules(args.target_modules)
            peft_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

        train_dataset = Dataset.from_generator(
            lambda: _grpo_prompt_records(snapshot, manifest["model"], input_adapter),
            cache_dir=str(work_dir / "dataset-cache"),
            keep_in_memory=False,
            num_proc=1,
            fingerprint=dataset_identity["sha256"][len(SHA256_PREFIX) :],
        )
        if len(train_dataset) != summary["row_count"]:
            raise ContractError("materialized GRPO prompt count differs from the verified corpus")
        replay = _RecordedRolloutSource(
            snapshot,
            dataset_identity,
            manifest["model"],
            input_adapter,
            num_generations,
        )

        class RecordedGRPOTrainer(GRPOTrainer):
            def _generate_and_score_completions(self, inputs):
                output = super()._generate_and_score_completions(inputs)
                recorded = output.get("sampling_per_token_logps")
                if recorded is None:
                    raise ContractError(
                        "TRL discarded the recorded behavior-policy log-probabilities"
                    )
                output["old_per_token_logps"] = recorded.detach()
                return output

        per_device_batch = num_generations * args.batch_size
        training_args = GRPOConfig(
            output_dir=str(work_dir / "trainer"),
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_generations=num_generations,
            max_completion_length=summary["max_completion_tokens"],
            shuffle_dataset=False,
            remove_unused_columns=False,
            steps_per_generation=1,
            num_iterations=1,
            use_vllm=False,
            beta=beta,
            epsilon=epsilon,
            epsilon_high=epsilon_high,
            loss_type=args.loss_type,
            importance_sampling_level=args.importance_sampling_level,
            scale_rewards=args.scale_rewards,
            mask_truncated_completions=False,
            use_bias_correction_kl=False,
            disable_dropout=True,
            seed=args.seed,
            data_seed=args.seed,
            bf16=bf16,
            fp16=fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs=(
                {"use_reentrant": False} if args.gradient_checkpointing else None
            ),
            optim="adamw_torch",
            lr_scheduler_type=args.lr_scheduler,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        )
        trainer = RecordedGRPOTrainer(
            model=model,
            reward_funcs=_recorded_reward,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            rollout_func=replay,
        )
        trainer.train()
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero():
            if replay.groups_served == 0:
                raise ContractError("TRL completed without consuming any recorded GRPO group")
            adapter_dir = work_dir / "adapter"
            trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
            effective: dict[str, dict[str, Any]] = {
                "base_model": _config_value("text", str(base_model)),
                "behavior_policy": _config_value("text", "recorded"),
                "behavior_policy_sha256": _config_value(
                    "text", _canonical_sha256(summary["behavior_policy"])
                ),
                "beta": _config_value("decimal", args.beta),
                "bf16": _config_value("boolean", bf16),
                "dataset_completions": _config_value(
                    "unsigned", summary["completion_count"]
                ),
                "dataset_forced_action_tokens": _config_value(
                    "unsigned", summary["forced_action_tokens"]
                ),
                "dataset_groups": _config_value("unsigned", summary["row_count"]),
                "dataset_sampled_action_tokens": _config_value(
                    "unsigned", summary["sampled_action_tokens"]
                ),
                "dtype": _config_value("text", dtype_name),
                "epochs": _config_value("decimal", args.epochs),
                "epsilon_high": _config_value("decimal", epsilon_high_text),
                "epsilon_low": _config_value("decimal", args.epsilon),
                "fp16": _config_value("boolean", fp16),
                "gradient_accumulation_steps": _config_value(
                    "unsigned", args.gradient_accumulation_steps
                ),
                "gradient_checkpointing": _config_value(
                    "boolean", args.gradient_checkpointing
                ),
                "groups_per_device_batch": _config_value("unsigned", args.batch_size),
                "importance_sampling_level": _config_value(
                    "text", args.importance_sampling_level
                ),
                "input_adapter": _config_value("boolean", input_adapter is not None),
                "kl_estimator": _config_value("text", "k3"),
                "kl_reference_policy": _config_value(
                    "text",
                    (
                        "none"
                        if beta == 0.0
                        else "initial_adapter"
                        if input_adapter is not None
                        else "base_model"
                    ),
                ),
                "learning_rate": _config_value("decimal", args.learning_rate),
                "loss_type": _config_value("text", args.loss_type),
                "lr_scheduler_type": _config_value("text", args.lr_scheduler),
                "max_completion_length": _config_value(
                    "unsigned", summary["max_completion_tokens"]
                ),
                "max_grad_norm": _config_value("decimal", args.max_grad_norm),
                "num_generations": _config_value("unsigned", num_generations),
                "num_iterations": _config_value("unsigned", 1),
                "optimizer": _config_value("text", "adamw_torch"),
                "per_device_train_batch_size": _config_value(
                    "unsigned", per_device_batch
                ),
                "scale_rewards": _config_value("text", args.scale_rewards),
                "seed": _config_value("unsigned", args.seed),
                "shuffle_dataset": _config_value("boolean", False),
                "steps_per_generation": _config_value("unsigned", 1),
                "use_bias_correction_kl": _config_value("boolean", False),
                "warmup_ratio": _config_value("decimal", args.warmup_ratio),
                "weight_decay": _config_value("decimal", args.weight_decay),
            }
            if input_adapter is None:
                effective.update(
                    {
                        "lora_alpha": _config_value("unsigned", alpha),
                        "lora_dropout": _config_value("decimal", dropout_text),
                        "lora_rank": _config_value("unsigned", rank),
                        "target_modules": _config_value(
                            "text", ",".join(target_modules)
                        ),
                    }
                )
            _publish_result(root, manifest, adapter_dir, effective)
    finally:
        if args.keep_work_dir:
            print(f"work directory retained at {work_dir}", file=sys.stderr)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


def run_training(args: argparse.Namespace, root: Path, manifest: dict[str, Any], base_model: Path) -> None:
    if not _script_matches_export(root, manifest) and not args.allow_custom_script:
        raise ContractError("executed script differs from exported train.py; pass --allow-custom-script to make that explicit")
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ContractError("WORLD_SIZE must be an integer") from exc
    if world_size != 1:
        raise ContractError(
            "the pinned v1 reference runner is single-process; use an explicit custom distributed "
            "script and --allow-custom-script so its executed bytes remain auditable"
        )
    _recover_or_reject_result_state(root)
    _assert_installed_pins()

    if manifest["task"] == "grpo":
        _run_grpo_training(args, root, manifest, base_model)
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    learning_rate = _decimal(args.learning_rate, "learning rate", minimum=0.0)
    epochs = _decimal(args.epochs, "epochs", minimum=0.0)
    if learning_rate <= 0.0 or epochs <= 0.0:
        raise ContractError("learning rate and epochs must be greater than zero")
    weight_decay = _decimal(args.weight_decay, "weight decay", minimum=0.0)
    warmup_ratio = _decimal(args.warmup_ratio, "warmup ratio", minimum=0.0, maximum=1.0)
    max_grad_norm = _decimal(args.max_grad_norm, "max grad norm", minimum=0.0)
    rows = _load_dataset_rows(root / "train.jsonl")
    normalized_rows = [{"messages": _normalize_messages_for_hf(row["messages"])} for row in rows]

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ContractError("pinned SFT route requires the exported fast tokenizer")
    tokenizer.chat_template = (root / "training_chat_template.jinja").read_text(encoding="utf-8")
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ContractError("base tokenizer defines neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    observed_max = 0
    for index, row in enumerate(normalized_rows, 1):
        encoded = tokenizer.apply_chat_template(
            row["messages"],
            chat_template=tokenizer.chat_template,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_ids = encoded["input_ids"]
        mask = encoded.get("assistant_masks")
        if not mask or not any(mask):
            raise ContractError(f"SFT row {index} has no assistant-supervised tokens in the exported template")
        observed_max = max(observed_max, len(input_ids))
    max_length = args.max_length or observed_max
    if max_length < observed_max:
        raise ContractError(
            f"--max-length={max_length} would truncate an admitted row of {observed_max} tokens; "
            "Kiln's reference route refuses silent corpus changes"
        )

    if args.dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype_name, dtype = "bfloat16", torch.bfloat16
        elif torch.cuda.is_available() or torch.backends.mps.is_available():
            dtype_name, dtype = "float16", torch.float16
        else:
            dtype_name, dtype = "float32", torch.float32
    else:
        dtype_name = args.dtype
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    bf16 = dtype_name == "bfloat16"
    fp16 = dtype_name == "float16"

    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
    )
    model.config.use_cache = not args.gradient_checkpointing

    input_adapter = manifest.get("input_adapter")
    peft_config = None
    if input_adapter is not None:
        if any(value is not None for value in (args.lora_rank, args.lora_alpha, args.lora_dropout, args.target_modules)):
            raise ContractError("LoRA creation options cannot be combined with an exported input adapter")
        model = PeftModel.from_pretrained(model, str(root / "input_adapter"), is_trainable=True)
    else:
        rank = args.lora_rank or 16
        alpha = args.lora_alpha or (rank * 2)
        dropout_text = args.lora_dropout or "0.0"
        dropout = _decimal(dropout_text, "LoRA dropout", minimum=0.0, maximum=1.0)
        target_modules = _target_modules(args.target_modules)
        peft_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    work_parent = root.parent
    work_dir = Path(tempfile.mkdtemp(prefix=f".{root.name}.sft-work-", dir=work_parent))
    try:
        training_args = SFTConfig(
            output_dir=str(work_dir / "trainer"),
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=max_length,
            assistant_only_loss=True,
            completion_only_loss=False,
            chat_template_path=str(root / "training_chat_template.jinja"),
            packing=False,
            shuffle_dataset=False,
            dataset_num_proc=1,
            seed=args.seed,
            data_seed=args.seed,
            bf16=bf16,
            fp16=fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
            optim="adamw_torch",
            lr_scheduler_type=args.lr_scheduler,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=Dataset.from_list(normalized_rows),
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        trainer.train()
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero():
            adapter_dir = work_dir / "adapter"
            trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
            effective: dict[str, dict[str, Any]] = {
                "assistant_only_loss": _config_value("boolean", True),
                "base_model": _config_value("text", str(base_model)),
                "bf16": _config_value("boolean", bf16),
                "dataset_rows": _config_value("unsigned", len(rows)),
                "dtype": _config_value("text", dtype_name),
                "epochs": _config_value("decimal", args.epochs),
                "fp16": _config_value("boolean", fp16),
                "gradient_accumulation_steps": _config_value("unsigned", args.gradient_accumulation_steps),
                "gradient_checkpointing": _config_value("boolean", args.gradient_checkpointing),
                "input_adapter": _config_value("boolean", input_adapter is not None),
                "learning_rate": _config_value("decimal", args.learning_rate),
                "lr_scheduler_type": _config_value("text", args.lr_scheduler),
                "max_grad_norm": _config_value("decimal", args.max_grad_norm),
                "max_length": _config_value("unsigned", max_length),
                "optimizer": _config_value("text", "adamw_torch"),
                "packing": _config_value("boolean", False),
                "per_device_train_batch_size": _config_value("unsigned", args.batch_size),
                "seed": _config_value("unsigned", args.seed),
                "shuffle_dataset": _config_value("boolean", False),
                "warmup_ratio": _config_value("decimal", args.warmup_ratio),
                "weight_decay": _config_value("decimal", args.weight_decay),
            }
            if input_adapter is None:
                effective.update(
                    {
                        "lora_alpha": _config_value("unsigned", alpha),
                        "lora_dropout": _config_value("decimal", dropout_text),
                        "lora_rank": _config_value("unsigned", rank),
                        "target_modules": _config_value("text", ",".join(target_modules)),
                    }
                )
            _publish_result(root, manifest, adapter_dir, effective)
    finally:
        if args.keep_work_dir:
            print(f"work directory retained at {work_dir}", file=sys.stderr)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="writable .kiln-hf export copy")
    parser.add_argument("--base-model", required=True, type=Path, help="local HF model directory whose shards match the export")
    parser.add_argument("--verify-only", action="store_true", help="validate bundle and base bytes without importing torch")
    parser.add_argument("--allow-custom-script", action="store_true", help="record, rather than reject, an executed script that differs from exported train.py")
    parser.add_argument("--learning-rate", default="2e-5")
    parser.add_argument("--epochs", default="1.0")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="SFT examples or GRPO prompt groups per device step",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--lr-scheduler", choices=("constant", "linear", "cosine"), default="linear")
    parser.add_argument("--warmup-ratio", default="0.0")
    parser.add_argument("--weight-decay", default="0.0")
    parser.add_argument("--max-grad-norm", default="1.0")
    parser.add_argument(
        "--beta",
        default="0.1",
        help="GRPO-only K3 KL coefficient",
    )
    parser.add_argument(
        "--epsilon",
        default="0.2",
        help="GRPO-only lower PPO clipping epsilon",
    )
    parser.add_argument(
        "--epsilon-high",
        help="GRPO-only upper PPO epsilon (or absolute CISPO cap)",
    )
    parser.add_argument(
        "--loss-type",
        choices=("grpo", "dapo", "bnpo", "dr_grpo", "cispo"),
        default="dapo",
        help="GRPO-only TRL loss aggregation",
    )
    parser.add_argument(
        "--importance-sampling-level",
        choices=("token", "sequence"),
        default="token",
        help="GRPO-only recorded-policy ratio aggregation",
    )
    parser.add_argument(
        "--scale-rewards",
        choices=("group", "batch", "none"),
        default="none",
        help="GRPO-only reward normalization",
    )
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout")
    parser.add_argument(
        "--target-modules",
        help="comma-separated Kiln-loadable LoRA modules; defaults to Kiln's complete supported set",
    )
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--keep-work-dir", action="store_true", help="retain temporary trainer outputs for diagnosis")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
            raise ContractError("batch size and gradient accumulation must be positive")
        if args.max_length is not None and args.max_length <= 0:
            raise ContractError("max length must be positive")
        if args.seed < 0 or args.seed > 2**32 - 1:
            raise ContractError("seed must fit the pinned trainer's unsigned 32-bit range")
        if args.lora_rank is not None and args.lora_rank <= 0:
            raise ContractError("LoRA rank must be positive")
        if args.lora_alpha is not None and args.lora_alpha <= 0:
            raise ContractError("LoRA alpha must be positive")
        root, manifest = load_export_bundle(args.bundle)
        base_model = verify_base_model_source(args.base_model, manifest)
        reference_match = _script_matches_export(root, manifest)
        if args.verify_only:
            print(
                json.dumps(
                    {
                        "status": "verified",
                        "export_sha256": manifest["export_sha256"],
                        "rows": manifest["data"]["row_count"],
                        "reference_script_match": reference_match,
                    },
                    sort_keys=True,
                )
            )
            return 0
        run_training(args, root, manifest, base_model)
        print(json.dumps({"status": "complete", "result": str(root / RESULT_MANIFEST)}, sort_keys=True))
        return 0
    except ContractError as exc:
        print(f"kiln HF/TRL contract error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("kiln HF/TRL training interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
