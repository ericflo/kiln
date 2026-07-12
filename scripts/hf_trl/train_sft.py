#!/usr/bin/env python3
"""Train one Kiln SFT handoff with the pinned HF/TRL/PEFT stack.

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
import sys
import tempfile
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
        data["sft_selection"]["ingestion_receipt"]["relative_path"],
        manifest["reference_script"]["relative_path"],
        manifest["environment_lock"]["relative_path"],
    }
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
    if manifest["task"] != "sft":
        raise ContractError("train_sft.py requires an SFT export")
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
    if not isinstance(base["shards"], list) or not base["shards"]:
        raise ContractError("base-weight manifest contains no shards")
    prior = ""
    total = 0
    for index, raw_shard in enumerate(base["shards"]):
        shard = _expect_keys(raw_shard, f"base shard {index}", {"filename", "size_bytes", "sha256"})
        filename = shard["filename"]
        if (
            not isinstance(filename, str)
            or PurePosixPath(filename).name != filename
            or not filename.endswith(".safetensors")
            or filename <= prior
        ):
            raise ContractError(f"invalid or unsorted base shard filename {filename!r}")
        size = shard["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ContractError(f"base shard {filename!r} has invalid size")
        _validate_sha256(shard["sha256"], f"base shard {filename!r} hash")
        total += size
        prior = filename
    if base["total_size_bytes"] != total:
        raise ContractError("base-weight total_size_bytes differs from its shards")
    _validate_sha256(base["aggregate_sha256"], "base-weight aggregate digest")

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

    data = _expect_keys(
        manifest["data"],
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
    if data["format"] != "sft_messages_jsonl" or not isinstance(data["row_count"], int) or data["row_count"] <= 0:
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
    if data["row_count"] != selection["rows_kept"] or data["ordered_corpus_sha256"] != selection["kept_corpus_sha256"]:
        raise ContractError("SFT data identity differs from its selection receipt")
    _verify_identity(root, data["dataset"], "train.jsonl")
    ingestion_path = _verify_identity(root, selection["ingestion_receipt"], "sft_ingestion.json")
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
    for field in ("invalid_row_policy", "rows_read", "rows_kept", "rows_rejected", "kept_corpus_sha256"):
        if ingestion[field] != selection[field]:
            raise ContractError(f"SFT selection field {field} differs from its receipt")
    if ingestion["source"] != data["source_name"]:
        raise ContractError("SFT data source differs from its ingestion receipt")
    if len(ingestion["kept_row_hashes"]) != ingestion["rows_kept"] or len(ingestion["rejected_rows"]) != ingestion["rows_rejected"]:
        raise ContractError("SFT ingestion receipt row evidence has the wrong length")
    if "split_manifest" in data:
        _verify_identity(root, data["split_manifest"], "split_manifest.json")
    _verify_identity(root, manifest["reference_script"], "train.py")
    _verify_identity(root, manifest["environment_lock"], "requirements.lock")

    adapter = manifest.get("input_adapter")
    if adapter is not None:
        adapter = _expect_keys(
            adapter,
            "input adapter",
            {"name", "config", "model"},
            {"kiln_manifest"},
        )
        _verify_identity(root, adapter["config"], "input_adapter/adapter_config.json")
        _verify_identity(root, adapter["model"], "input_adapter/adapter_model.safetensors")
        if "kiln_manifest" in adapter:
            _verify_identity(root, adapter["kiln_manifest"], "input_adapter/adapter_manifest.json")

    _declared_export_files(root, manifest)
    rows = _load_dataset_rows(_bundle_file(root, "train.jsonl"))
    if len(rows) != data["row_count"]:
        raise ContractError(f"train.jsonl has {len(rows)} rows; manifest declares {data['row_count']}")
    template = _bundle_file(root, "training_chat_template.jinja").read_text(encoding="utf-8")
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
        trainer = {
            "kind": "trl_sft_trainer",
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
            "task": "sft",
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
        target_modules = args.target_modules or "all-linear"
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
                        "target_modules": _config_value("text", target_modules),
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--lr-scheduler", choices=("constant", "linear", "cosine"), default="linear")
    parser.add_argument("--warmup-ratio", default="0.0")
    parser.add_argument("--weight-decay", default="0.0")
    parser.add_argument("--max-grad-norm", default="1.0")
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout")
    parser.add_argument("--target-modules", help="PEFT target module selector; defaults to all-linear")
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
        print(f"kiln HF/TRL SFT contract error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("kiln HF/TRL SFT interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
