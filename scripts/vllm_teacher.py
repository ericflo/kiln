#!/usr/bin/env python3
"""Fingerprint and launch an immutable vLLM prompt-logprob teacher.

The launcher owns every identity-bearing vLLM option.  Arbitrary additional
options are accepted only after ``--`` and must use unambiguous ``--key=value``
form (with a small allowlist of valueless boolean switches).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import struct
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qualification import model_fingerprint as _model_fingerprint  # noqa: E402


IDENTITY_SCHEMA = "kiln.teacher-identity.v1"
PROTOCOL = "vllm.prompt-logprobs.numeric-token-ids.causal.v1"
LOGPROBS_MODE = "raw_logprobs"
INPUT_MANIFEST_SCHEMA = "kiln.vllm-teacher-input.v1"
INFERENCE_CONFIG_SCHEMA = "kiln.vllm-inference-config.v1"
FINGERPRINT_PREFIX = "kiln-teacher-v1"
TOKENIZER_VOCAB_DOMAIN = b"kiln.tokenizer-vocab.v1\0"
BASE_MODEL_DOMAIN = b"kiln.base-model-content.v1\0"
ADAPTER_WEIGHTS_DOMAIN = b"kiln.adapter-weights.v1\0"
MIN_VLLM_VERSION = (0, 25, 0)
MAX_IDENTITY_JSON_BYTES = 4 * 1024
MAX_FINGERPRINT_BYTES = 6 * 1024
MAX_NAME_BYTES = 256
MAX_IMPLEMENTATION_BYTES = 256
MAX_VOCAB_SIZE = 16_777_216
MAX_TOP_K = 65_536
MAX_MODEL_LEN = 16_777_216

IDENTITY_FIELDS = (
    "schema",
    "protocol",
    "served_model_id",
    "base_model_sha256",
    "tokenizer_vocab_sha256",
    "tokenizer_config_sha256",
    "adapter",
    "vocab_size",
    "max_top_k",
    "max_model_len",
    "logprobs_mode",
    "implementation",
    "inference_config_sha256",
)
ADAPTER_FIELDS = ("name", "weights_sha256", "config_sha256")
INPUT_MANIFEST_FIELDS = (
    "schema",
    "base_model_sha256",
    "model_config_sha256",
    "tokenizer_vocab_sha256",
    "tokenizer_config_sha256",
    "adapter",
    "adapter_max_rank",
    "vocab_size",
    "implementation",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
OPTION_RE = re.compile(r"^--[a-z0-9][a-z0-9-]*$")
VERSION_RE = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?(?:[A-Za-z0-9.+-]*)$")

# These options would replace an input or contract owned by this launcher.
FORBIDDEN_OPTIONS = {
    "--model",
    "--served-model-name",
    "--max-model-len",
    "--max-logprobs",
    "--logprobs-mode",
    "--fingerprint-mode",
    "--fingerprint-value",
    "--middleware",
    "--tokenizer",
    "--tokenizer-mode",
    "--tokenizer-revision",
    "--revision",
    "--code-revision",
    "--generation-config",
    "--trust-remote-code",
    "--runner",
    "--task",
    "--grpc",
    "--tokens-only",
    "--help",
    "--version",
}

# vLLM uses valueless switches for these common options. Everything else must
# use --key=value so a value can never be mistaken for a second model path.
VALUELESS_OPTIONS = {
    "--aggregate-engine-logging",
    "--disable-cascade-attn",
    "--disable-custom-all-reduce",
    "--disable-fastapi-docs",
    "--disable-frontend-multiprocessing",
    "--disable-log-stats",
    "--disable-uvicorn-access-log",
    "--enable-offline-docs",
    "--enable-prefix-caching",
    "--enable-request-id-headers",
    "--enforce-eager",
    "--fail-on-environ-validation",
}

# Transport changes do not alter teacher logits and therefore do not invalidate
# a logit cache. They remain in the command but are excluded from this digest.
TRANSPORT_OPTIONS = {
    "--host",
    "--port",
    "--uds",
    "--api-key",
    "--root-path",
    "--uvicorn-log-level",
    "--disable-uvicorn-access-log",
    "--enable-request-id-headers",
    "--disable-fastapi-docs",
    "--enable-offline-docs",
    "--allowed-origins",
    "--allowed-methods",
    "--allowed-headers",
    "--allow-credentials",
    "--h11-max-incomplete-event-size",
    "--h11-max-header-count",
}
TRANSPORT_PREFIXES = ("--ssl-", "--disable-access-log-")
INFERENCE_ENV_PREFIXES = (
    "VLLM_",
    "CUDA_",
    "HIP_",
    "ROCM_",
    "HSA_",
    "NCCL_",
    "TORCH_",
    "PYTORCH_",
)


class TeacherLaunchError(RuntimeError):
    """Raised when an immutable teacher cannot be described or launched."""


def _reject_constant(value: str) -> None:
    raise TeacherLaunchError(f"non-finite JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TeacherLaunchError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json_object(payload: bytes, source: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TeacherLaunchError(f"{source} is not valid UTF-8: {exc}") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except TeacherLaunchError:
        raise
    except json.JSONDecodeError as exc:
        raise TeacherLaunchError(f"failed to parse {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise TeacherLaunchError(f"{source} must contain a JSON object")
    return value


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _raw_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise TeacherLaunchError(f"{field} must be exactly 64 lowercase hexadecimal characters")
    return value


def _strip_qualified_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise TeacherLaunchError(f"{field} is not a qualification SHA-256 value")
    return _raw_sha256(value[len("sha256:") :], field)


def _feed_bytes(digest: "hashlib._Hash", payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _feed_hash(digest: "hashlib._Hash", value: str, field: str) -> None:
    digest.update(bytes.fromhex(_raw_sha256(value, field)))


def _normal_directory(path: Path, label: str) -> Path:
    if "\x00" in os.fspath(path):
        raise TeacherLaunchError(f"{label} contains a NUL byte")
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        info = absolute.lstat()
    except OSError as exc:
        raise TeacherLaunchError(f"cannot inspect {label} {absolute}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise TeacherLaunchError(f"{label} must not be a symlink: {absolute}")
    if not stat.S_ISDIR(info.st_mode):
        raise TeacherLaunchError(f"{label} is not a directory: {absolute}")
    return absolute


def _hash_regular_file(root: Path, relative: str, label: str) -> tuple[str, int]:
    try:
        opened = _model_fingerprint._open_regular(root, relative)
    except _model_fingerprint.ModelFingerprintError as exc:
        raise TeacherLaunchError(f"cannot fingerprint {label}: {exc}") from exc
    try:
        digest = _strip_qualified_sha256(opened.hash(), label)
        _verify_opened_unchanged(opened, label)
        return digest, opened.initial_stat.st_size
    finally:
        opened.close()


def _verify_opened_unchanged(opened: Any, label: str) -> None:
    try:
        descriptor_after = os.fstat(opened.fd)
        path_after = opened.path.stat(follow_symlinks=False)
    except OSError as exc:
        raise TeacherLaunchError(f"{label} changed while it was being read: {exc}") from exc
    initial = _model_fingerprint._stat_identity(opened.initial_stat)
    if (
        _model_fingerprint._stat_identity(descriptor_after) != initial
        or _model_fingerprint._stat_identity(path_after) != initial
    ):
        raise TeacherLaunchError(f"{label} changed while it was being read")


def fingerprint_base_model_details(model_path: Path) -> tuple[str, str]:
    """Return the Rust-loader-compatible weight digest and model-config digest."""

    root = _normal_directory(model_path, "model path")
    try:
        model = _model_fingerprint.fingerprint_model(root)
    except _model_fingerprint.ModelFingerprintError as exc:
        raise TeacherLaunchError(f"base-model fingerprint failed: {exc}") from exc

    weights = model.get("weight_files")
    if not isinstance(weights, list) or not weights:
        raise TeacherLaunchError("qualification model fingerprint returned no weights")

    records: list[tuple[bytes, int]] = []
    for index, item in enumerate(weights):
        if not isinstance(item, dict):
            raise TeacherLaunchError(f"qualification weight_files[{index}] is not an object")
        byte_count = item.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise TeacherLaunchError(f"qualification weight_files[{index}].bytes is invalid")
        content_hash = _strip_qualified_sha256(
            item.get("sha256"), f"weight_files[{index}].sha256"
        )
        records.append((bytes.fromhex(content_hash), byte_count))

    records.sort()
    digest = hashlib.sha256()
    digest.update(BASE_MODEL_DOMAIN)
    digest.update(struct.pack("<Q", len(records)))
    for content_hash, byte_count in records:
        digest.update(struct.pack("<Q", byte_count))
        digest.update(content_hash)
    config_hash = _strip_qualified_sha256(model.get("config_hash"), "config_hash")
    return digest.hexdigest(), config_hash


def fingerprint_base_model(model_path: Path) -> str:
    return fingerprint_base_model_details(model_path)[0]


def tokenizer_config_fingerprint(backend_tokenizer_json: str) -> str:
    if not isinstance(backend_tokenizer_json, str) or not backend_tokenizer_json:
        raise TeacherLaunchError("backend tokenizer JSON must be a non-empty string")
    try:
        payload = backend_tokenizer_json.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TeacherLaunchError("backend tokenizer JSON is not valid UTF-8") from exc
    _strict_json_object(payload, "backend tokenizer JSON")
    return _sha256_hex(payload)


def tokenizer_vocab_fingerprint(vocab: Mapping[str, int]) -> tuple[str, int]:
    """Hash token ID semantics in a form shared with the Rust tokenizer."""

    if not isinstance(vocab, Mapping) or not vocab:
        raise TeacherLaunchError("tokenizer vocabulary must be a non-empty mapping")
    entries: list[tuple[int, bytes]] = []
    for token, token_id in vocab.items():
        if not isinstance(token, str):
            raise TeacherLaunchError("tokenizer vocabulary keys must be strings")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TeacherLaunchError(f"token ID for {token!r} must be an integer")
        if token_id < 0 or token_id > 0xFFFF_FFFF:
            raise TeacherLaunchError(f"token ID for {token!r} is outside the u32 range")
        try:
            raw = token.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise TeacherLaunchError(f"token {token!r} is not valid UTF-8") from exc
        entries.append((token_id, raw))

    entries.sort(key=lambda entry: (entry[0], entry[1]))
    digest = hashlib.sha256()
    digest.update(TOKENIZER_VOCAB_DOMAIN)
    digest.update(struct.pack("<Q", len(entries)))
    for token_id, raw in entries:
        digest.update(struct.pack("<I", token_id))
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest(), len(entries)


def _load_tokenizer_contract(model_path: Path) -> tuple[Mapping[str, int], str, int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TeacherLaunchError(
            "transformers is required to fingerprint a real tokenizer; install it in the "
            "vLLM environment or use --identity-input with --manifest-only/--dry-run"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            os.fspath(model_path),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        vocab = tokenizer.get_vocab()
        backend = tokenizer.backend_tokenizer
        backend_json = backend.to_str()
        vocab_size = backend.get_vocab_size(with_added_tokens=True)
    except Exception as exc:
        raise TeacherLaunchError(f"failed to load the local tokenizer contract: {exc}") from exc
    if not isinstance(vocab, Mapping):
        raise TeacherLaunchError("transformers tokenizer get_vocab() did not return a mapping")
    if not isinstance(backend_json, str) or not backend_json:
        raise TeacherLaunchError("transformers backend_tokenizer.to_str() returned no JSON")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise TeacherLaunchError("backend tokenizer returned an invalid vocabulary size")
    return vocab, backend_json, vocab_size


def fingerprint_adapter(adapter_path: Path, name: str) -> dict[str, Any]:
    root = _normal_directory(adapter_path, "adapter path")
    try:
        config_input = _model_fingerprint._open_regular(root, "adapter_config.json")
    except _model_fingerprint.ModelFingerprintError as exc:
        raise TeacherLaunchError(f"cannot fingerprint adapter_config.json: {exc}") from exc
    try:
        config_payload = config_input.read_bytes()
        config_hash = _strip_qualified_sha256(config_input.hash(), "adapter_config.json")
        _strict_json_object(config_payload, "adapter_config.json")
        _verify_opened_unchanged(config_input, "adapter_config.json")
    finally:
        config_input.close()

    candidates: list[str] = []
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        try:
            if _model_fingerprint._path_exists_without_following(root, filename):
                candidates.append(filename)
        except _model_fingerprint.ModelFingerprintError as exc:
            raise TeacherLaunchError(str(exc)) from exc
    if not candidates:
        raise TeacherLaunchError(
            "adapter path must contain adapter_model.safetensors or adapter_model.bin"
        )
    if len(candidates) != 1:
        raise TeacherLaunchError(
            "adapter path contains both safetensors and bin weights; the loader input is ambiguous"
        )
    filename = candidates[0]
    weight_hash, byte_count = _hash_regular_file(root, filename, filename)
    if byte_count <= 0:
        raise TeacherLaunchError(f"adapter weight file is empty: {filename}")

    aggregate = hashlib.sha256()
    aggregate.update(ADAPTER_WEIGHTS_DOMAIN)
    aggregate.update(struct.pack("<Q", 1))
    _feed_bytes(aggregate, filename.encode("utf-8"))
    aggregate.update(struct.pack("<Q", byte_count))
    _feed_hash(aggregate, weight_hash, filename)
    return {
        "name": name,
        "weights_sha256": aggregate.hexdigest(),
        "config_sha256": config_hash,
    }


def adapter_max_rank(adapter_path: Path) -> int:
    root = _normal_directory(adapter_path, "adapter path")
    try:
        config_input = _model_fingerprint._open_regular(root, "adapter_config.json")
    except _model_fingerprint.ModelFingerprintError as exc:
        raise TeacherLaunchError(f"cannot inspect adapter rank: {exc}") from exc
    try:
        config = _strict_json_object(config_input.read_bytes(), "adapter_config.json")
        _verify_opened_unchanged(config_input, "adapter_config.json")
    finally:
        config_input.close()

    ranks = [config.get("r")]
    rank_pattern = config.get("rank_pattern", {})
    if rank_pattern is None:
        rank_pattern = {}
    if not isinstance(rank_pattern, dict):
        raise TeacherLaunchError("adapter_config.json rank_pattern must be an object")
    ranks.extend(rank_pattern.values())
    for rank in ranks:
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise TeacherLaunchError("adapter_config.json ranks must be positive integers")
    required = max(ranks)
    for supported in (1, 8, 16, 32, 64, 128, 256, 320, 512):
        if required <= supported:
            return supported
    raise TeacherLaunchError("adapter rank exceeds vLLM's supported maximum of 512")


def _version_tuple(version: str) -> tuple[int, int, int]:
    match = VERSION_RE.fullmatch(version)
    if match is None:
        raise TeacherLaunchError(f"vLLM version is not safely parseable: {version!r}")
    return tuple(int(value or "0") for value in match.groups())  # type: ignore[return-value]


def _validate_implementation(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("vllm:"):
        raise TeacherLaunchError("implementation must have the form vllm:<version>")
    version = value[len("vllm:") :]
    if len(value.encode("utf-8")) > MAX_IMPLEMENTATION_BYTES:
        raise TeacherLaunchError("implementation exceeds the 256-byte identity limit")
    if _version_tuple(version) < MIN_VLLM_VERSION:
        raise TeacherLaunchError("immutable custom fingerprints require vLLM 0.25.0 or newer")
    return value


def _installed_vllm_version() -> str:
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TeacherLaunchError("vLLM is not installed in this Python environment") from exc
    _validate_implementation(f"vllm:{version}")
    return version


def validate_extra_vllm_args(args: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in args:
        if not isinstance(raw, str) or not raw:
            raise TeacherLaunchError("vLLM arguments must be non-empty strings")
        if "\x00" in raw or "\n" in raw or "\r" in raw:
            raise TeacherLaunchError("vLLM arguments must not contain NUL or newline characters")
        option, separator, value = raw.partition("=")
        if not OPTION_RE.fullmatch(option):
            raise TeacherLaunchError(
                f"ambiguous vLLM argument {raw!r}; use one long --key=value argument"
            )
        if option in seen:
            raise TeacherLaunchError(f"duplicate vLLM option is not allowed: {option}")
        seen.add(option)
        if option in FORBIDDEN_OPTIONS or "lora" in option or "adapter" in option:
            raise TeacherLaunchError(f"the immutable launcher owns or forbids {option}")
        if not separator and option not in VALUELESS_OPTIONS:
            raise TeacherLaunchError(f"vLLM option must use --key=value form: {option}")
        if separator and not value:
            raise TeacherLaunchError(f"vLLM option value must not be empty: {option}")
        if option == "--load-format" and value not in {"auto", "safetensors"}:
            raise TeacherLaunchError(
                "--load-format must be auto or safetensors so the fingerprint matches loaded weights"
            )
        result.append(raw)
    return result


def inference_config_fingerprint(
    *,
    model_config_sha256: str,
    max_top_k: int,
    max_model_len: int,
    adapter_enabled: bool,
    adapter_max_rank: int | None,
    extra_args: Sequence[str],
    environment: Mapping[str, str],
) -> str:
    if adapter_enabled != (adapter_max_rank is not None):
        raise TeacherLaunchError("adapter mode and adapter_max_rank are inconsistent")
    if adapter_max_rank is not None and (
        isinstance(adapter_max_rank, bool)
        or not isinstance(adapter_max_rank, int)
        or adapter_max_rank not in {1, 8, 16, 32, 64, 128, 256, 320, 512}
    ):
        raise TeacherLaunchError("adapter_max_rank is not supported by vLLM")
    validated = validate_extra_vllm_args(extra_args)
    inference_args = []
    for raw in validated:
        option = raw.partition("=")[0]
        if option in TRANSPORT_OPTIONS or option.startswith(TRANSPORT_PREFIXES):
            continue
        inference_args.append(raw)
    inference_args.sort(key=lambda value: value.encode("utf-8"))
    inference_environment = {
        key: value
        for key, value in sorted(environment.items())
        if key.startswith(INFERENCE_ENV_PREFIXES)
        and key not in {"VLLM_ALLOW_RUNTIME_LORA_UPDATING"}
    }
    value = {
        "schema": INFERENCE_CONFIG_SCHEMA,
        "model_config_sha256": _raw_sha256(
            model_config_sha256, "model_config_sha256"
        ),
        "max_top_k": max_top_k,
        "max_model_len": max_model_len,
        "logprobs_mode": LOGPROBS_MODE,
        "generation_config": "vllm",
        "adapter_enabled": adapter_enabled,
        "adapter_max_rank": adapter_max_rank,
        "runtime_lora_updates": False,
        "vllm_args": inference_args,
        "environment": inference_environment,
    }
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_hex(payload)


def _validate_adapter(value: Any, *, served_model_id: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or tuple(value) != ADAPTER_FIELDS:
        raise TeacherLaunchError(
            "adapter must be null or an ordered object with name, weights_sha256, config_sha256"
        )
    name = value.get("name")
    if name != served_model_id:
        raise TeacherLaunchError("static adapter name must equal served_model_id")
    return {
        "name": name,
        "weights_sha256": _raw_sha256(value.get("weights_sha256"), "adapter.weights_sha256"),
        "config_sha256": _raw_sha256(value.get("config_sha256"), "adapter.config_sha256"),
    }


def build_identity(
    *,
    served_model_id: str,
    base_model_sha256: str,
    tokenizer_vocab_sha256: str,
    tokenizer_config_sha256: str,
    adapter: dict[str, Any] | None,
    vocab_size: int,
    max_top_k: int,
    max_model_len: int,
    implementation: str,
    inference_config_sha256: str,
) -> dict[str, Any]:
    if (
        not isinstance(served_model_id, str)
        or not MODEL_ID_RE.fullmatch(served_model_id)
        or len(served_model_id.encode("utf-8")) > MAX_NAME_BYTES
    ):
        raise TeacherLaunchError(
            "served_model_id must be 1-256 ASCII model-name characters and may not contain spaces"
        )
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or not 1 <= vocab_size <= MAX_VOCAB_SIZE
    ):
        raise TeacherLaunchError(f"vocab_size must be in 1..={MAX_VOCAB_SIZE}")
    if (
        isinstance(max_top_k, bool)
        or not isinstance(max_top_k, int)
        or max_top_k <= 0
        or max_top_k > vocab_size
        or max_top_k > MAX_TOP_K
    ):
        raise TeacherLaunchError("max_top_k must be in 1..=vocab_size")
    if (
        isinstance(max_model_len, bool)
        or not isinstance(max_model_len, int)
        or max_model_len <= 0
        or max_model_len > MAX_MODEL_LEN
    ):
        raise TeacherLaunchError(f"max_model_len must be in 1..={MAX_MODEL_LEN}")
    adapter_value = _validate_adapter(adapter, served_model_id=served_model_id)
    identity = {
        "schema": IDENTITY_SCHEMA,
        "protocol": PROTOCOL,
        "served_model_id": served_model_id,
        "base_model_sha256": _raw_sha256(base_model_sha256, "base_model_sha256"),
        "tokenizer_vocab_sha256": _raw_sha256(
            tokenizer_vocab_sha256, "tokenizer_vocab_sha256"
        ),
        "tokenizer_config_sha256": _raw_sha256(
            tokenizer_config_sha256, "tokenizer_config_sha256"
        ),
        "adapter": adapter_value,
        "vocab_size": vocab_size,
        "max_top_k": max_top_k,
        "max_model_len": max_model_len,
        "logprobs_mode": LOGPROBS_MODE,
        "implementation": _validate_implementation(implementation),
        "inference_config_sha256": _raw_sha256(
            inference_config_sha256, "inference_config_sha256"
        ),
    }
    return identity


def canonical_identity_json(identity: Mapping[str, Any]) -> bytes:
    if not isinstance(identity, dict) or tuple(identity) != IDENTITY_FIELDS:
        raise TeacherLaunchError(
            "TeacherIdentityV1 keys are missing, extra, or not in canonical field order"
        )
    rebuilt = build_identity(
        served_model_id=identity["served_model_id"],
        base_model_sha256=identity["base_model_sha256"],
        tokenizer_vocab_sha256=identity["tokenizer_vocab_sha256"],
        tokenizer_config_sha256=identity["tokenizer_config_sha256"],
        adapter=identity["adapter"],
        vocab_size=identity["vocab_size"],
        max_top_k=identity["max_top_k"],
        max_model_len=identity["max_model_len"],
        implementation=identity["implementation"],
        inference_config_sha256=identity["inference_config_sha256"],
    )
    if rebuilt != identity:
        raise TeacherLaunchError("TeacherIdentityV1 values are not canonical")
    payload = json.dumps(
        identity,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > MAX_IDENTITY_JSON_BYTES:
        raise TeacherLaunchError("TeacherIdentityV1 exceeds the 4096-byte canonical JSON limit")
    return payload


def encode_system_fingerprint(identity: Mapping[str, Any]) -> str:
    payload = canonical_identity_json(identity)
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    fingerprint = f"{FINGERPRINT_PREFIX}.{encoded}.{_sha256_hex(payload)}"
    if len(fingerprint.encode("ascii")) > MAX_FINGERPRINT_BYTES:
        raise TeacherLaunchError("teacher system fingerprint exceeds the 6144-byte limit")
    return fingerprint


def decode_system_fingerprint(value: str) -> dict[str, Any]:
    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_FINGERPRINT_BYTES:
        raise TeacherLaunchError("teacher system fingerprint is missing or too large")
    parts = value.split(".")
    if len(parts) != 3 or parts[0] != FINGERPRINT_PREFIX:
        raise TeacherLaunchError("teacher system fingerprint has the wrong prefix or shape")
    encoded, claimed_hash = parts[1], parts[2]
    _raw_sha256(claimed_hash, "system fingerprint digest")
    if not encoded or "=" in encoded or not re.fullmatch(r"[A-Za-z0-9_-]+", encoded):
        raise TeacherLaunchError("teacher identity payload is not unpadded base64url")
    try:
        payload = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (ValueError, base64.binascii.Error) as exc:
        raise TeacherLaunchError("teacher identity payload is not valid base64url") from exc
    if base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=") != encoded:
        raise TeacherLaunchError("teacher identity payload does not use canonical base64url")
    if _sha256_hex(payload) != claimed_hash:
        raise TeacherLaunchError("teacher identity payload digest does not match")
    identity = _strict_json_object(payload, "teacher identity payload")
    if tuple(identity) != IDENTITY_FIELDS:
        raise TeacherLaunchError("teacher identity fields are not in canonical order")
    if canonical_identity_json(identity) != payload:
        raise TeacherLaunchError("teacher identity JSON is not canonical")
    return identity


def load_identity_input(path: Path) -> dict[str, Any]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        info = absolute.lstat()
    except OSError as exc:
        raise TeacherLaunchError(f"cannot inspect identity input {absolute}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise TeacherLaunchError(f"identity input must not be a symlink: {absolute}")
    if not stat.S_ISREG(info.st_mode):
        raise TeacherLaunchError(f"identity input is not a regular file: {absolute}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(absolute, flags)
    except OSError as exc:
        raise TeacherLaunchError(f"cannot open identity input {absolute}: {exc}") from exc
    try:
        before = os.fstat(fd)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        path_after = absolute.stat(follow_symlinks=False)
    finally:
        os.close(fd)
    identity_tuple = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity_tuple(before) != identity_tuple(after) or identity_tuple(before) != identity_tuple(
        path_after
    ):
        raise TeacherLaunchError("identity input changed while it was being read")
    value = _strict_json_object(b"".join(chunks), os.fspath(absolute))
    if set(value) != set(INPUT_MANIFEST_FIELDS):
        missing = sorted(set(INPUT_MANIFEST_FIELDS) - set(value))
        extra = sorted(set(value) - set(INPUT_MANIFEST_FIELDS))
        raise TeacherLaunchError(f"identity input has missing keys {missing} and extra keys {extra}")
    if value.get("schema") != INPUT_MANIFEST_SCHEMA:
        raise TeacherLaunchError(f"identity input schema must be {INPUT_MANIFEST_SCHEMA!r}")
    vocab_size = value.get("vocab_size")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise TeacherLaunchError("identity input vocab_size must be a positive integer")
    adapter = value.get("adapter")
    if adapter is not None:
        if not isinstance(adapter, dict) or set(adapter) != set(ADAPTER_FIELDS):
            raise TeacherLaunchError("identity input adapter has the wrong fields")
        adapter = {
            "name": adapter.get("name"),
            "weights_sha256": _raw_sha256(
                adapter.get("weights_sha256"), "adapter.weights_sha256"
            ),
            "config_sha256": _raw_sha256(adapter.get("config_sha256"), "adapter.config_sha256"),
        }
    adapter_rank = value.get("adapter_max_rank")
    if (adapter is None) != (adapter_rank is None):
        raise TeacherLaunchError(
            "identity input adapter and adapter_max_rank must both be null or both be present"
        )
    if adapter_rank is not None and (
        isinstance(adapter_rank, bool)
        or not isinstance(adapter_rank, int)
        or adapter_rank not in {1, 8, 16, 32, 64, 128, 256, 320, 512}
    ):
        raise TeacherLaunchError("identity input adapter_max_rank is not supported by vLLM")
    return {
        "base_model_sha256": _raw_sha256(
            value.get("base_model_sha256"), "base_model_sha256"
        ),
        "model_config_sha256": _raw_sha256(
            value.get("model_config_sha256"), "model_config_sha256"
        ),
        "tokenizer_vocab_sha256": _raw_sha256(
            value.get("tokenizer_vocab_sha256"), "tokenizer_vocab_sha256"
        ),
        "tokenizer_config_sha256": _raw_sha256(
            value.get("tokenizer_config_sha256"), "tokenizer_config_sha256"
        ),
        "adapter": adapter,
        "adapter_max_rank": adapter_rank,
        "vocab_size": vocab_size,
        "implementation": _validate_implementation(value.get("implementation")),
    }


def build_vllm_command(
    *,
    model_path: Path,
    served_model_id: str,
    adapter_path: Path | None,
    adapter_max_rank: int | None,
    max_top_k: int,
    max_model_len: int,
    system_fingerprint: str,
    extra_args: Sequence[str],
) -> list[str]:
    model_root = _normal_directory(model_path, "model path")
    adapter_root = (
        _normal_directory(adapter_path, "adapter path") if adapter_path is not None else None
    )
    if (adapter_root is None) != (adapter_max_rank is None):
        raise TeacherLaunchError("adapter path and adapter_max_rank are inconsistent")
    if adapter_max_rank is not None and (
        isinstance(adapter_max_rank, bool)
        or adapter_max_rank not in {1, 8, 16, 32, 64, 128, 256, 320, 512}
    ):
        raise TeacherLaunchError("adapter_max_rank is not supported by vLLM")
    validated = validate_extra_vllm_args(extra_args)
    base_name = served_model_id
    if adapter_root is not None:
        base_name = f"kiln-base-{decode_system_fingerprint(system_fingerprint)['base_model_sha256'][:16]}"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        os.fspath(model_root),
        f"--served-model-name={base_name}",
        f"--max-model-len={max_model_len}",
        f"--max-logprobs={max_top_k}",
        f"--logprobs-mode={LOGPROBS_MODE}",
        "--generation-config=vllm",
        "--fingerprint-mode=custom",
        f"--fingerprint-value={system_fingerprint}",
    ]
    if adapter_root is not None:
        module = json.dumps(
            {
                "name": served_model_id,
                "path": os.fspath(adapter_root),
                "base_model_name": base_name,
            },
            separators=(",", ":"),
        )
        command.extend(
            [
                "--enable-lora",
                "--max-loras=1",
                "--max-cpu-loras=1",
                f"--max-lora-rank={adapter_max_rank}",
                f"--lora-modules={module}",
            ]
        )
    command.extend(validated)
    return command


def launch_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "0"
    environment.pop("VLLM_LORA_RESOLVER_CACHE_DIR", None)
    return environment


def validate_launch_environment(environment: Mapping[str, str]) -> None:
    plugins = environment.get("VLLM_PLUGINS", "").strip()
    if plugins:
        raise TeacherLaunchError(
            "VLLM_PLUGINS must be unset; resolver plugins can mutate the served model set"
        )
    if environment.get("VLLM_SKIP_MODEL_NAME_VALIDATION", "").strip().lower() in {
        "1",
        "true",
    }:
        raise TeacherLaunchError(
            "VLLM_SKIP_MODEL_NAME_VALIDATION must be disabled for an identity-bound teacher"
        )
    for key, value in environment.items():
        if (
            key.startswith("VLLM_")
            and "LORA" in key
            and key not in {"VLLM_ALLOW_RUNTIME_LORA_UPDATING", "VLLM_LORA_RESOLVER_CACHE_DIR"}
            and value
        ):
            raise TeacherLaunchError(f"unsupported LoRA-affecting environment variable: {key}")


def _redact_command(command: Sequence[str]) -> list[str]:
    return ["--api-key=<redacted>" if item.startswith("--api-key=") else item for item in command]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="local base model directory")
    parser.add_argument("--served-model-id", required=True, help="the only accepted API model ID")
    parser.add_argument("--adapter-path", type=Path, help="one immutable static LoRA adapter")
    parser.add_argument("--max-top-k", required=True, type=int, help="maximum prompt_logprobs K")
    parser.add_argument("--max-model-len", required=True, type=int, help="maximum token context")
    parser.add_argument(
        "--identity-input",
        type=Path,
        help="strict precomputed inputs; allowed only for non-launch test/dry-run modes",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--manifest-only", action="store_true", help="emit identity JSON and exit")
    mode.add_argument("--dry-run", action="store_true", help="emit identity and redacted argv JSON")
    parser.add_argument(
        "vllm_args",
        nargs=argparse.REMAINDER,
        help="additional vLLM options after --, each in --key=value form",
    )
    args = parser.parse_args(list(argv))
    if args.vllm_args and args.vllm_args[0] == "--":
        args.vllm_args = args.vllm_args[1:]
    return args


def _identity_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if args.identity_input is not None:
        if not (args.manifest_only or args.dry_run):
            raise TeacherLaunchError("--identity-input is forbidden for a real launch")
        return load_identity_input(args.identity_input)
    if args.model_path is None:
        raise TeacherLaunchError("--model-path is required without --identity-input")
    model_path = _normal_directory(args.model_path, "model path")
    base_hash, model_config_hash = fingerprint_base_model_details(model_path)
    vocab, backend_tokenizer_json, vocab_size = _load_tokenizer_contract(model_path)
    tokenizer_config_hash = tokenizer_config_fingerprint(backend_tokenizer_json)
    vocab_hash, _pair_count = tokenizer_vocab_fingerprint(vocab)
    if args.adapter_path is not None:
        adapter = fingerprint_adapter(args.adapter_path, args.served_model_id)
        max_adapter_rank = adapter_max_rank(args.adapter_path)
    else:
        adapter = None
        max_adapter_rank = None
    return {
        "base_model_sha256": base_hash,
        "model_config_sha256": model_config_hash,
        "tokenizer_vocab_sha256": vocab_hash,
        "tokenizer_config_sha256": tokenizer_config_hash,
        "adapter": adapter,
        "adapter_max_rank": max_adapter_rank,
        "vocab_size": vocab_size,
        "implementation": f"vllm:{_installed_vllm_version()}",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if not args.manifest_only:
            validate_launch_environment(os.environ)
        runtime_environment = launch_environment()
        extra_args = validate_extra_vllm_args(args.vllm_args)
        inputs = _identity_inputs(args)
        adapter = inputs["adapter"]
        max_adapter_rank = inputs["adapter_max_rank"]
        if adapter is not None and adapter.get("name") != args.served_model_id:
            raise TeacherLaunchError("identity input adapter name must equal --served-model-id")
        if args.identity_input is not None:
            if args.dry_run and args.model_path is None:
                raise TeacherLaunchError("--model-path is required for --dry-run command output")
            if args.adapter_path is not None and adapter is None:
                raise TeacherLaunchError("--adapter-path conflicts with the null manifest adapter")
            if args.dry_run and adapter is not None and args.adapter_path is None:
                raise TeacherLaunchError(
                    "--adapter-path presence must match the identity input adapter field"
                )
        inference_hash = inference_config_fingerprint(
            model_config_sha256=inputs["model_config_sha256"],
            max_top_k=args.max_top_k,
            max_model_len=args.max_model_len,
            adapter_enabled=adapter is not None,
            adapter_max_rank=max_adapter_rank,
            extra_args=extra_args,
            environment=runtime_environment,
        )
        identity = build_identity(
            served_model_id=args.served_model_id,
            base_model_sha256=inputs["base_model_sha256"],
            tokenizer_vocab_sha256=inputs["tokenizer_vocab_sha256"],
            tokenizer_config_sha256=inputs["tokenizer_config_sha256"],
            adapter=adapter,
            vocab_size=inputs["vocab_size"],
            max_top_k=args.max_top_k,
            max_model_len=args.max_model_len,
            implementation=inputs["implementation"],
            inference_config_sha256=inference_hash,
        )
        fingerprint = encode_system_fingerprint(identity)
        output: dict[str, Any] = {
            "identity": identity,
            "canonical_json": canonical_identity_json(identity).decode("utf-8"),
            "system_fingerprint": fingerprint,
        }
        command: list[str] | None = None
        if args.dry_run or not args.manifest_only:
            if args.model_path is None:
                raise TeacherLaunchError("--model-path is required to build a vLLM command")
            command = build_vllm_command(
                model_path=args.model_path,
                served_model_id=args.served_model_id,
                adapter_path=args.adapter_path,
                adapter_max_rank=max_adapter_rank,
                max_top_k=args.max_top_k,
                max_model_len=args.max_model_len,
                system_fingerprint=fingerprint,
                extra_args=extra_args,
            )
        if args.dry_run and command is not None:
            output["command"] = _redact_command(command)
            output["runtime_lora_updates"] = "disabled"
        if args.manifest_only or args.dry_run:
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0

        assert command is not None
        print(json.dumps({"system_fingerprint": fingerprint}), flush=True)
        os.execve(sys.executable, command, runtime_environment)
        raise AssertionError("os.execve returned unexpectedly")
    except TeacherLaunchError as exc:
        print(f"vLLM teacher launch failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
