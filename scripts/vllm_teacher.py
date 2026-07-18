#!/usr/bin/env python3
"""Fingerprint and launch an immutable vLLM prompt-logprob teacher.

The launcher owns every identity-bearing vLLM option.  Arbitrary additional
options are accepted only after ``--`` and must use unambiguous ``--key=value``
form (with a small allowlist of valueless boolean switches).
"""

from __future__ import annotations

import argparse
import base64
import errno
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import secrets
import signal
import stat
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qualification import model_fingerprint as _model_fingerprint  # noqa: E402


IDENTITY_SCHEMA = "kiln.teacher-identity.v1"
PROTOCOL = "vllm.prompt-logprobs.numeric-token-ids.causal.v1"
LOGPROBS_MODE = "raw_logprobs"
INPUT_MANIFEST_SCHEMA = "kiln.vllm-teacher-input.v2"
INFERENCE_CONFIG_SCHEMA = "kiln.vllm-inference-config.v2"
FINGERPRINT_PREFIX = "kiln-teacher-v1"
TOKENIZER_VOCAB_DOMAIN = b"kiln.tokenizer-vocab.v1\0"
BASE_MODEL_DOMAIN = b"kiln.base-model-content.v1\0"
ADAPTER_WEIGHTS_DOMAIN = b"kiln.adapter-weights.v1\0"
# Custom OpenAI response fingerprints shipped in upstream vLLM v0.20.1rc0.
# Treat prerelease suffixes as part of the corresponding three-component build;
# the required behavior is revalidated in the fresh child before every launch.
MIN_VLLM_VERSION = (0, 20, 1)
MIN_VLLM_VERSION_TEXT = "0.20.1rc0"
MAX_IDENTITY_JSON_BYTES = 4 * 1024
MAX_FINGERPRINT_BYTES = 6 * 1024
MAX_INPUT_MANIFEST_BYTES = 64 * 1024
MAX_NAME_BYTES = 256
MAX_IMPLEMENTATION_BYTES = 256
MAX_VOCAB_SIZE = 16_777_216
MAX_TOP_K = 65_536
MAX_MODEL_LEN = 16_777_216
MAX_PROMPT_LOGPROB_CANDIDATES = 1_000_000
DEFAULT_PROMPT_LOGPROB_CANDIDATES = 1_000_000
SNAPSHOT_SCHEMA = "kiln.vllm-teacher-snapshot.v1"
SNAPSHOT_MANIFEST = "snapshot-manifest.json"
COPY_CHUNK_BYTES = 8 * 1024 * 1024
FICLONE = 0x40049409
MAX_SNAPSHOT_FILES = 100_000
MAX_SNAPSHOT_DIRECTORIES = 100_000
MAX_SNAPSHOT_BYTES = 16 * 1024**4
MAX_SNAPSHOT_DEPTH = 128
MAX_SNAPSHOT_PATH_BYTES = 4096
MAX_SNAPSHOT_PATH_METADATA_BYTES = 16 * 1024**2
MAX_SNAPSHOT_MANIFEST_BYTES = 128 * 1024**2
SNAPSHOT_HEADROOM_BYTES = 64 * 1024**2
PROCESS_GROUP_TERM_SECONDS = 10.0
PROCESS_GROUP_KILL_SECONDS = 2.0
PROCESS_GROUP_MODE_DETACHED = "detached"
PROCESS_GROUP_MODE_INHERITED = "inherited"
PROCESS_GROUP_MODES = (PROCESS_GROUP_MODE_DETACHED, PROCESS_GROUP_MODE_INHERITED)
RUNTIME_CONTENT_SCHEMA = "kiln.python-runtime-content.v1"
RUNTIME_CONTENT_DOMAIN = b"kiln.python-runtime-content.v1\0"
RUNTIME_PACKAGES = ("vllm", "torch", "transformers", "tokenizers")
MAX_RUNTIME_CONTENT_FILES = 250_000
MAX_RUNTIME_CONTENT_DIRECTORIES = 100_000
MAX_RUNTIME_CONTENT_BYTES = 64 * 1024**3
MAX_RUNTIME_CONTENT_DEPTH = 128
MAX_RUNTIME_CONTENT_PATH_BYTES = 4096
MAX_RUNTIME_CONTENT_PATH_METADATA_BYTES = 64 * 1024**2
RUNTIME_CONTENT_PROBE_TIMEOUT_SECONDS = 600

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
    "max_prompt_logprob_candidates",
    "logprobs_mode",
    "implementation",
    "inference_config_sha256",
)
ADAPTER_FIELDS = ("name", "weights_sha256", "config_sha256")
INPUT_MANIFEST_FIELDS = (
    "schema",
    "base_model_sha256",
    "snapshot_content_sha256",
    "model_config_sha256",
    "tokenizer_vocab_sha256",
    "tokenizer_config_sha256",
    "adapter",
    "adapter_max_rank",
    "vocab_size",
    "implementation",
    "runtime_versions",
    "runtime_content_sha256",
    "accelerator",
)
RUNTIME_VERSION_FIELDS = (
    "python",
    "python_implementation",
    "vllm",
    "torch",
    "transformers",
    "tokenizers",
)
RUNTIME_PROBE_FIELDS = (
    "schema",
    "runtime_versions",
    "runtime_content_sha256",
    "file_count",
    "directory_count",
    "logical_bytes",
)
ACCELERATOR_FIELDS = ("type", "driver", "devices")
ACCELERATOR_DEVICE_FIELDS = ("index", "name", "architecture", "total_memory_bytes")
ACCELERATOR_TYPES = {"cpu", "cuda", "rocm", "metal", "xpu"}
MAX_RUNTIME_VALUE_BYTES = 512
MAX_ACCELERATOR_DEVICES = 256
DETERMINISM_ENV_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDNN_DETERMINISTIC",
    "HIPBLASLT_TUNING_OVERRIDE_FILE",
    "NVIDIA_TF32_OVERRIDE",
    "PYTHONHASHSEED",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
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
    "--load-format",
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
    "--language-model-only",
}

# Every inference-affecting pass-through option is explicitly reviewed. Unknown
# vLLM options fail closed so a new model/code/file input cannot silently evade
# the identity contract.
VETTED_INFERENCE_OPTIONS = {
    "--all2all-backend",
    "--attention-backend",
    "--block-size",
    "--cpu-offload-gb",
    "--cudagraph-capture-sizes",
    "--data-parallel-size",
    "--device",
    "--dtype",
    "--gpu-memory-utilization",
    "--kv-cache-dtype",
    "--max-cudagraph-capture-size",
    "--max-num-batched-tokens",
    "--max-num-seqs",
    "--max-seq-len-to-capture",
    "--num-gpu-blocks-override",
    "--pipeline-parallel-size",
    "--quantization",
    "--rope-scaling",
    "--rope-theta",
    "--scheduling-policy",
    "--seed",
    "--swap-space",
    "--tensor-parallel-size",
}

# vLLM also accepts importable class paths for attention backends. Keep this a
# closed value set so adding the option cannot introduce unbound executable
# code beneath an otherwise unchanged runtime identity.
REVIEWED_ATTENTION_BACKENDS = frozenset({"TRITON_ATTN"})

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
    "ROCR_",
    "HSA_",
    "NCCL_",
    "RCCL_",
    "TORCH_",
    "PYTORCH_",
    "TRITON_",
    "HF_",
    "OMP_",
    "MKL_",
    "ONEAPI_",
    "ZE_",
)
INFERENCE_ENV_KEYS = {
    "GPU_DEVICE_ORDINAL",
    "PATH",
}
FORBIDDEN_FILE_ENV_KEYS = {
    "CC",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "CUDAHOSTCXX",
    "CUDACXX",
    "CXX",
    "C_INCLUDE_PATH",
    "HIPBLASLT_TUNING_OVERRIDE_FILE",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "LIBRARY_PATH",
    "NVCC",
    "PYTHONHOME",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "PYTHONSAFEPATH",
    "XDG_CACHE_HOME",
}
UNBOUND_ENV_NAME_PARTS = {
    "CLASS",
    "COMMAND",
    "DIR",
    "EXECUTABLE",
    "FILE",
    "HOME",
    "MODULE",
    "PATH",
    "PLUGIN",
    "PLUGINS",
    "ROOT",
}
SAFE_ENV_NAME_EXCEPTIONS = {"CUDA_MODULE_LOADING", "PATH"}
PATH_LIKE_INFERENCE_OPTION_PARTS = {"dir", "directory", "file", "path", "template"}


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


@dataclass(frozen=True, order=True)
class SnapshotFile:
    relative_path: str
    byte_len: int
    sha256: str


@dataclass
class SnapshotBudget:
    files: int = 0
    directories: int = 0
    bytes: int = 0
    path_bytes: int = 0
    expected_files: int | None = None
    expected_directories: int | None = None
    expected_bytes: int | None = None
    expected_path_bytes: int | None = None

    def add_directory(self, relative_path: str, depth: int) -> None:
        _validate_snapshot_relative_path(relative_path, depth)
        self.directories += 1
        self.path_bytes += len(relative_path.encode("utf-8"))
        if self.directories > MAX_SNAPSHOT_DIRECTORIES:
            raise TeacherLaunchError(
                f"snapshot exceeds the {MAX_SNAPSHOT_DIRECTORIES} directory safety limit"
            )
        if self.expected_directories is not None and self.directories > self.expected_directories:
            raise TeacherLaunchError("snapshot source added directories after inventory")
        if self.expected_path_bytes is not None and self.path_bytes > self.expected_path_bytes:
            raise TeacherLaunchError("snapshot source paths grew after inventory")
        if self.path_bytes > MAX_SNAPSHOT_PATH_METADATA_BYTES:
            raise TeacherLaunchError("snapshot path metadata exceeds its aggregate safety limit")

    def add_file(self, relative_path: str, byte_len: int, depth: int) -> None:
        _validate_snapshot_relative_path(relative_path, depth)
        if byte_len < 0:
            raise TeacherLaunchError(f"snapshot file has a negative size: {relative_path!r}")
        self.files += 1
        self.bytes += byte_len
        self.path_bytes += len(relative_path.encode("utf-8"))
        if self.files > MAX_SNAPSHOT_FILES:
            raise TeacherLaunchError(
                f"snapshot exceeds the {MAX_SNAPSHOT_FILES} regular-file safety limit"
            )
        if self.bytes > MAX_SNAPSHOT_BYTES:
            raise TeacherLaunchError(
                f"snapshot exceeds the {MAX_SNAPSHOT_BYTES}-byte logical-size safety limit"
            )
        if self.expected_files is not None and self.files > self.expected_files:
            raise TeacherLaunchError("snapshot source added files after inventory")
        if self.expected_bytes is not None and self.bytes > self.expected_bytes:
            raise TeacherLaunchError("snapshot source grew after the capacity check")
        if self.expected_path_bytes is not None and self.path_bytes > self.expected_path_bytes:
            raise TeacherLaunchError("snapshot source paths grew after inventory")
        if self.path_bytes > MAX_SNAPSHOT_PATH_METADATA_BYTES:
            raise TeacherLaunchError("snapshot path metadata exceeds its aggregate safety limit")


@dataclass
class RuntimeContentBudget:
    files: int = 0
    directories: int = 0
    logical_bytes: int = 0
    path_bytes: int = 0

    def add_directory(self, label: str, depth: int) -> None:
        self._add_path(label, depth)
        self.directories += 1
        if self.directories > MAX_RUNTIME_CONTENT_DIRECTORIES:
            raise TeacherLaunchError(
                "Python runtime content exceeds the "
                f"{MAX_RUNTIME_CONTENT_DIRECTORIES}-directory safety limit"
            )

    def add_file(self, label: str, byte_len: int, depth: int) -> None:
        self._add_path(label, depth)
        if byte_len < 0:
            raise TeacherLaunchError(f"Python runtime file has a negative size: {label!r}")
        self.files += 1
        self.logical_bytes += byte_len
        if self.files > MAX_RUNTIME_CONTENT_FILES:
            raise TeacherLaunchError(
                "Python runtime content exceeds the "
                f"{MAX_RUNTIME_CONTENT_FILES}-file safety limit"
            )
        if self.logical_bytes > MAX_RUNTIME_CONTENT_BYTES:
            raise TeacherLaunchError(
                "Python runtime content exceeds the "
                f"{MAX_RUNTIME_CONTENT_BYTES}-byte logical-size safety limit"
            )

    def _add_path(self, label: str, depth: int) -> None:
        if depth > MAX_RUNTIME_CONTENT_DEPTH:
            raise TeacherLaunchError(
                "Python runtime content exceeds the "
                f"{MAX_RUNTIME_CONTENT_DEPTH}-directory nesting limit"
            )
        if not label or any(ord(character) < 0x20 or ord(character) == 0x7F for character in label):
            raise TeacherLaunchError(f"Python runtime content has an invalid path label: {label!r}")
        try:
            encoded = label.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise TeacherLaunchError(
                f"Python runtime content path is not valid UTF-8: {label!r}"
            ) from exc
        if len(encoded) > MAX_RUNTIME_CONTENT_PATH_BYTES:
            raise TeacherLaunchError(
                "Python runtime content path exceeds the "
                f"{MAX_RUNTIME_CONTENT_PATH_BYTES}-byte safety limit"
            )
        self.path_bytes += len(encoded)
        if self.path_bytes > MAX_RUNTIME_CONTENT_PATH_METADATA_BYTES:
            raise TeacherLaunchError(
                "Python runtime path metadata exceeds its aggregate safety limit"
            )


@dataclass(frozen=True, order=True)
class RuntimeContentRecord:
    label: str
    byte_len: int
    sha256: str


@dataclass(frozen=True)
class RuntimeContentSnapshot:
    sha256: str
    python_executable: Path
    file_count: int
    directory_count: int
    logical_bytes: int


@dataclass
class ImmutableSnapshot:
    snapshot_root: Path
    path: Path
    model_path: Path
    adapter_path: Path | None
    directories: tuple[str, ...]
    files: tuple[SnapshotFile, ...]
    manifest_sha256: str
    _root_fd: int
    _root_identity: tuple[Any, ...]
    _snapshot_identity: tuple[Any, ...]
    _snapshot_device: int
    _cleaned: bool = False

    def verify(self) -> None:
        """Re-hash the complete published snapshot and reject any mutation."""

        try:
            root_descriptor = os.fstat(self._root_fd)
            root_path = self.snapshot_root.stat(follow_symlinks=False)
            snapshot_entry = os.stat(
                self.path.name,
                dir_fd=self._root_fd,
                follow_symlinks=False,
            )
            snapshot_path = self.path.stat(follow_symlinks=False)
        except OSError as exc:
            raise TeacherLaunchError(f"cannot verify snapshot path anchors: {exc}") from exc
        if (
            _directory_anchor_identity(root_descriptor) != self._root_identity
            or _directory_anchor_identity(root_path) != self._root_identity
        ):
            raise TeacherLaunchError("snapshot root was replaced after publication")
        if (
            _model_fingerprint._stat_identity(snapshot_entry) != self._snapshot_identity
            or _model_fingerprint._stat_identity(snapshot_path) != self._snapshot_identity
        ):
            raise TeacherLaunchError("published snapshot directory was replaced")
        root = _normal_directory(self.path, "published snapshot")
        _require_read_only(root, "published snapshot")
        expected_root_entries = {"model", SNAPSHOT_MANIFEST}
        if self.adapter_path is not None:
            expected_root_entries.add("adapter")
        actual_root_entries = _directory_entry_names(root, "published snapshot")
        if actual_root_entries != expected_root_entries:
            raise TeacherLaunchError(
                "published snapshot root changed: expected "
                f"{sorted(expected_root_entries)!r}, observed {sorted(actual_root_entries)!r}"
            )

        manifest_hash, _ = _hash_snapshot_file(
            root / SNAPSHOT_MANIFEST,
            SNAPSHOT_MANIFEST,
            require_read_only=True,
        )
        if manifest_hash != self.manifest_sha256:
            raise TeacherLaunchError("published snapshot manifest digest changed")

        observed: list[SnapshotFile] = []
        observed_directories: list[str] = []
        budget = SnapshotBudget()
        _scan_snapshot_tree(
            root / "model",
            "model",
            observed,
            require_read_only=True,
            budget=budget,
            directories=observed_directories,
        )
        if self.adapter_path is not None:
            _scan_snapshot_tree(
                root / "adapter",
                "adapter",
                observed,
                require_read_only=True,
                budget=budget,
                directories=observed_directories,
            )
        observed.sort()
        observed_directories.sort(key=lambda value: value.encode("utf-8"))
        if tuple(observed_directories) != self.directories or tuple(observed) != self.files:
            raise TeacherLaunchError("published snapshot content does not match its manifest")

        manifest = _read_strict_regular_file(root / SNAPSHOT_MANIFEST, SNAPSHOT_MANIFEST)
        expected_manifest = _snapshot_manifest_bytes(
            self.files,
            self.adapter_path is not None,
            self.directories,
        )
        if manifest != expected_manifest:
            raise TeacherLaunchError("published snapshot manifest is not canonical")

    def cleanup(self) -> None:
        if self._cleaned:
            return
        _validate_snapshot_child(self.snapshot_root, self.path, "ready-")
        try:
            _remove_private_child(
                self._root_fd,
                self.path.name,
                expected_identity=self._snapshot_identity,
                expected_device=self._snapshot_device,
            )
        except OSError as exc:
            raise TeacherLaunchError(
                f"failed to remove immutable snapshot {self.path}: {exc}"
            ) from exc
        finally:
            if self._root_fd >= 0:
                os.close(self._root_fd)
                self._root_fd = -1
        self._cleaned = True


def default_snapshot_root() -> Path:
    configured = os.environ.get("KILN_VLLM_SNAPSHOT_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "kiln" / "teacher-snapshots"


def _assert_no_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as exc:
            raise TeacherLaunchError(f"cannot inspect {label} component {current}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise TeacherLaunchError(f"{label} must not traverse a symlink: {current}")


def _trusted_snapshot_directory_owner(info: os.stat_result) -> bool:
    return not hasattr(os, "getuid") or info.st_uid in {0, os.getuid()}


def _secure_snapshot_root(path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    if absolute == Path(absolute.anchor):
        raise TeacherLaunchError(
            "snapshot root must be a dedicated directory, not a filesystem root"
        )
    current = Path(absolute.anchor)
    final_created = False
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o700)
                info = current.lstat()
                if current == absolute:
                    final_created = True
            except OSError as exc:
                raise TeacherLaunchError(
                    f"cannot create snapshot root component {current}: {exc}"
                ) from exc
        except OSError as exc:
            raise TeacherLaunchError(
                f"cannot inspect snapshot root component {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise TeacherLaunchError(f"snapshot root must not traverse a symlink: {current}")
        if not stat.S_ISDIR(info.st_mode):
            raise TeacherLaunchError(f"snapshot root component is not a directory: {current}")
        if not _trusted_snapshot_directory_owner(info):
            raise TeacherLaunchError(
                f"snapshot root ancestry has an untrusted owner: {current}"
            )
        permissions = stat.S_IMODE(info.st_mode)
        if permissions & 0o022 and not permissions & stat.S_ISVTX:
            raise TeacherLaunchError(
                "snapshot root ancestry must not permit untrusted rename: "
                f"{current} has mode {permissions:04o}"
            )
    root = _normal_directory(absolute, "snapshot root")
    try:
        info = root.stat(follow_symlinks=False)
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise TeacherLaunchError(f"snapshot root is not owned by the current user: {root}")
        if stat.S_IMODE(info.st_mode) != 0o700:
            origin = "new" if final_created else "existing"
            raise TeacherLaunchError(
                f"{origin} snapshot root must have mode 0700 without permission mutation: {root}"
            )
    except OSError as exc:
        raise TeacherLaunchError(f"cannot verify snapshot root privacy: {exc}") from exc
    return root


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((os.fspath(path), os.fspath(parent))) == os.fspath(parent)
    except ValueError:
        return False


def _directory_anchor_identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid)


def _validate_snapshot_relative_path(relative_path: str, depth: int) -> None:
    if depth > MAX_SNAPSHOT_DEPTH:
        raise TeacherLaunchError(
            f"snapshot exceeds the {MAX_SNAPSHOT_DEPTH}-directory nesting limit"
        )
    try:
        byte_len = len(relative_path.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise TeacherLaunchError(f"snapshot path is not valid UTF-8: {relative_path!r}") from exc
    if byte_len > MAX_SNAPSHOT_PATH_BYTES:
        raise TeacherLaunchError(
            f"snapshot path exceeds the {MAX_SNAPSHOT_PATH_BYTES}-byte safety limit"
        )


def _directory_entry_names(path: Path, label: str) -> set[str]:
    try:
        return {entry.name for entry in os.scandir(path)}
    except OSError as exc:
        raise TeacherLaunchError(f"cannot enumerate {label} {path}: {exc}") from exc


def _inventory_source_tree(
    path: Path,
    logical_path: str,
    budget: SnapshotBudget,
    *,
    depth: int = 0,
) -> tuple[int, int]:
    budget.add_directory(logical_path, depth)
    try:
        entries = list(os.scandir(path))
    except OSError as exc:
        raise TeacherLaunchError(
            f"cannot inventory snapshot source {logical_path!r}: {exc}"
        ) from exc
    file_count = 0
    byte_count = 0
    for entry in entries:
        child_logical = f"{logical_path}/{entry.name}"
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise TeacherLaunchError(f"cannot inventory {child_logical!r}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise TeacherLaunchError(
                f"snapshot source must not contain symlinks: {child_logical!r}"
            )
        if stat.S_ISDIR(info.st_mode):
            child_files, child_bytes = _inventory_source_tree(
                Path(entry.path), child_logical, budget, depth=depth + 1
            )
            file_count += child_files
            byte_count += child_bytes
        elif stat.S_ISREG(info.st_mode):
            budget.add_file(child_logical, info.st_size, depth + 1)
            file_count += 1
            byte_count += info.st_size
        else:
            raise TeacherLaunchError(f"snapshot source contains a special file: {child_logical!r}")
    return file_count, byte_count


def _snapshot_filesystem_capacity(path: Path) -> tuple[int, int | None, int]:
    try:
        info = os.statvfs(path)
    except OSError as exc:
        raise TeacherLaunchError(f"cannot inspect snapshot filesystem capacity: {exc}") from exc
    block_size = info.f_frsize or info.f_bsize
    available_inodes = None if info.f_files == 0 else info.f_favail
    return info.f_bavail * block_size, available_inodes, block_size


def _hash_fd(fd: int, expected_len: int | None = None) -> tuple[str, int]:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    byte_len = 0
    while expected_len is None or byte_len < expected_len:
        read_len = COPY_CHUNK_BYTES
        if expected_len is not None:
            read_len = min(read_len, expected_len - byte_len)
        chunk = os.read(fd, read_len)
        if not chunk:
            if expected_len is not None and byte_len != expected_len:
                raise TeacherLaunchError("snapshot file became shorter while being read")
            break
        digest.update(chunk)
        byte_len += len(chunk)
    if expected_len is not None and os.read(fd, 1):
        raise TeacherLaunchError("snapshot file became larger while being read")
    return digest.hexdigest(), byte_len


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise TeacherLaunchError("snapshot copy made no write progress")
        offset += written


def _try_reflink(source_fd: int, destination_fd: int) -> bool:
    try:
        import fcntl
    except ImportError:
        return False
    try:
        fcntl.ioctl(destination_fd, FICLONE, source_fd)
        return True
    except OSError as exc:
        if exc.errno not in {
            errno.EXDEV,
            errno.EOPNOTSUPP,
            errno.ENOTTY,
            errno.EINVAL,
            errno.EPERM,
            errno.ENOSYS,
        }:
            raise TeacherLaunchError(f"reflink snapshot copy failed unexpectedly: {exc}") from exc
        os.ftruncate(destination_fd, 0)
        os.lseek(destination_fd, 0, os.SEEK_SET)
        return False


def _copy_regular_file(
    source: Path,
    destination: Path,
    relative_path: str,
    budget: SnapshotBudget,
    depth: int,
) -> SnapshotFile:
    source_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    destination_flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        source_fd = os.open(source, source_flags)
    except OSError as exc:
        raise TeacherLaunchError(f"cannot open snapshot source {relative_path!r}: {exc}") from exc
    destination_fd: int | None = None
    try:
        source_initial = os.fstat(source_fd)
        if not stat.S_ISREG(source_initial.st_mode):
            raise TeacherLaunchError(f"snapshot source is not a regular file: {relative_path!r}")
        budget.add_file(relative_path, source_initial.st_size, depth)
        destination_fd = os.open(destination, destination_flags, 0o600)
        if not _try_reflink(source_fd, destination_fd):
            os.lseek(source_fd, 0, os.SEEK_SET)
            remaining = source_initial.st_size
            while remaining:
                chunk = os.read(source_fd, min(COPY_CHUNK_BYTES, remaining))
                if not chunk:
                    raise TeacherLaunchError(
                        f"snapshot source became shorter while copied: {relative_path!r}"
                    )
                _write_all(destination_fd, chunk)
                remaining -= len(chunk)
            if os.read(source_fd, 1):
                raise TeacherLaunchError(
                    f"snapshot source became larger while copied: {relative_path!r}"
                )
        os.fsync(destination_fd)
        source_hash, source_len = _hash_fd(source_fd, source_initial.st_size)
        destination_hash, destination_len = _hash_fd(
            destination_fd, source_initial.st_size
        )
        if (source_hash, source_len) != (destination_hash, destination_len):
            raise TeacherLaunchError(f"snapshot copy mismatch for {relative_path!r}")
        source_after = os.fstat(source_fd)
        source_path_after = source.stat(follow_symlinks=False)
        identity = _model_fingerprint._stat_identity(source_initial)
        if (
            _model_fingerprint._stat_identity(source_after) != identity
            or _model_fingerprint._stat_identity(source_path_after) != identity
        ):
            raise TeacherLaunchError(f"snapshot source changed while copied: {relative_path!r}")
        destination_after = os.fstat(destination_fd)
        if not stat.S_ISREG(destination_after.st_mode) or destination_after.st_size != source_len:
            raise TeacherLaunchError(f"snapshot destination is invalid: {relative_path!r}")
        os.chmod(destination, 0o400, follow_symlinks=False)
        return SnapshotFile(relative_path, source_len, source_hash)
    except OSError as exc:
        raise TeacherLaunchError(f"snapshot copy failed for {relative_path!r}: {exc}") from exc
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)


def _copy_snapshot_tree(
    source: Path,
    destination: Path,
    logical_path: str,
    records: list[SnapshotFile],
    directories: list[str],
    budget: SnapshotBudget,
    *,
    depth: int = 0,
) -> None:
    try:
        source_info = source.lstat()
    except OSError as exc:
        raise TeacherLaunchError(f"cannot inspect snapshot source {logical_path!r}: {exc}") from exc
    if stat.S_ISLNK(source_info.st_mode):
        raise TeacherLaunchError(f"snapshot source must not contain symlinks: {logical_path!r}")
    if not stat.S_ISDIR(source_info.st_mode):
        raise TeacherLaunchError(f"snapshot source is not a directory: {logical_path!r}")
    try:
        source_fd = os.open(
            source,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise TeacherLaunchError(f"cannot open snapshot directory {logical_path!r}: {exc}") from exc
    try:
        source_initial = os.fstat(source_fd)
        budget.add_directory(logical_path, depth)
        directories.append(logical_path)
        try:
            destination.mkdir(mode=0o700)
        except OSError as exc:
            raise TeacherLaunchError(
                f"cannot create snapshot directory {logical_path!r}: {exc}"
            ) from exc
        try:
            entries = sorted(os.scandir(source), key=lambda entry: os.fsencode(entry.name))
        except OSError as exc:
            raise TeacherLaunchError(
                f"cannot enumerate snapshot source {logical_path!r}: {exc}"
            ) from exc
        for entry in entries:
            try:
                entry.name.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise TeacherLaunchError(
                    f"snapshot source name is not valid UTF-8 under {logical_path!r}"
                ) from exc
            child_logical = f"{logical_path}/{entry.name}"
            try:
                entry_info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise TeacherLaunchError(f"cannot inspect {child_logical!r}: {exc}") from exc
            source_child = source / entry.name
            destination_child = destination / entry.name
            if stat.S_ISLNK(entry_info.st_mode):
                raise TeacherLaunchError(
                    f"snapshot source must not contain symlinks: {child_logical!r}"
                )
            if stat.S_ISDIR(entry_info.st_mode):
                _copy_snapshot_tree(
                    source_child,
                    destination_child,
                    child_logical,
                    records,
                    directories,
                    budget,
                    depth=depth + 1,
                )
            elif stat.S_ISREG(entry_info.st_mode):
                records.append(
                    _copy_regular_file(
                        source_child,
                        destination_child,
                        child_logical,
                        budget,
                        depth + 1,
                    )
                )
            else:
                raise TeacherLaunchError(
                    f"snapshot source contains a special file: {child_logical!r}"
                )
        source_after = os.fstat(source_fd)
        source_path_after = source.stat(follow_symlinks=False)
        identity = _model_fingerprint._stat_identity(source_initial)
        if (
            _model_fingerprint._stat_identity(source_after) != identity
            or _model_fingerprint._stat_identity(source_path_after) != identity
        ):
            raise TeacherLaunchError(f"snapshot source directory changed: {logical_path!r}")
    finally:
        os.close(source_fd)


def _snapshot_manifest_bytes(
    files: Sequence[SnapshotFile],
    has_adapter: bool,
    directories: Sequence[str],
) -> bytes:
    value = {
        "schema": SNAPSHOT_SCHEMA,
        "has_adapter": has_adapter,
        "directories": list(directories),
        "files": [
            {
                "path": item.relative_path,
                "bytes": item.byte_len,
                "sha256": item.sha256,
            }
            for item in files
        ],
    }
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def snapshot_content_fingerprint(model_path: Path, adapter_path: Path | None) -> str:
    records: list[SnapshotFile] = []
    directories: list[str] = []
    budget = SnapshotBudget()
    _scan_snapshot_tree(
        _normal_directory(model_path, "model path"),
        "model",
        records,
        require_read_only=False,
        budget=budget,
        directories=directories,
    )
    if adapter_path is not None:
        _scan_snapshot_tree(
            _normal_directory(adapter_path, "adapter path"),
            "adapter",
            records,
            require_read_only=False,
            budget=budget,
            directories=directories,
        )
    records.sort()
    directories.sort(key=lambda value: value.encode("utf-8"))
    return _sha256_hex(
        _snapshot_manifest_bytes(records, adapter_path is not None, directories)
    )


def _write_snapshot_manifest(path: Path, payload: bytes) -> str:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(path, flags, 0o600)
        try:
            _write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(path, 0o400, follow_symlinks=False)
    except OSError as exc:
        raise TeacherLaunchError(f"cannot publish snapshot manifest: {exc}") from exc
    return _sha256_hex(payload)


def _freeze_snapshot_directories(path: Path) -> None:
    try:
        entries = list(os.scandir(path))
    except OSError as exc:
        raise TeacherLaunchError(f"cannot freeze snapshot directory {path}: {exc}") from exc
    for entry in entries:
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise TeacherLaunchError(f"snapshot unexpectedly contains symlink {entry.path}")
        if stat.S_ISDIR(info.st_mode):
            _freeze_snapshot_directories(Path(entry.path))
        elif not stat.S_ISREG(info.st_mode):
            raise TeacherLaunchError(f"snapshot unexpectedly contains special file {entry.path}")
    try:
        os.chmod(path, 0o500, follow_symlinks=False)
    except OSError as exc:
        raise TeacherLaunchError(f"cannot make snapshot directory read-only: {exc}") from exc


def _require_read_only(path: Path, label: str) -> None:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise TeacherLaunchError(f"cannot inspect {label}: {exc}") from exc
    if info.st_mode & 0o222:
        raise TeacherLaunchError(f"{label} is writable")


def _hash_snapshot_file(
    path: Path,
    relative_path: str,
    *,
    require_read_only: bool,
) -> tuple[str, int]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise TeacherLaunchError(f"cannot open snapshot file {relative_path!r}: {exc}") from exc
    try:
        initial = os.fstat(fd)
        if not stat.S_ISREG(initial.st_mode):
            raise TeacherLaunchError(f"snapshot entry is not a regular file: {relative_path!r}")
        if require_read_only and initial.st_mode & 0o222:
            raise TeacherLaunchError(f"snapshot file is writable: {relative_path!r}")
        digest, byte_len = _hash_fd(fd, initial.st_size)
        after = os.fstat(fd)
        path_after = path.stat(follow_symlinks=False)
        identity = _model_fingerprint._stat_identity(initial)
        if (
            _model_fingerprint._stat_identity(after) != identity
            or _model_fingerprint._stat_identity(path_after) != identity
        ):
            raise TeacherLaunchError(f"snapshot file changed while verified: {relative_path!r}")
        return digest, byte_len
    except OSError as exc:
        raise TeacherLaunchError(f"cannot verify snapshot file {relative_path!r}: {exc}") from exc
    finally:
        os.close(fd)


def _scan_snapshot_tree(
    path: Path,
    logical_path: str,
    records: list[SnapshotFile],
    *,
    require_read_only: bool,
    budget: SnapshotBudget | None = None,
    directories: list[str] | None = None,
    depth: int = 0,
) -> None:
    if budget is None:
        budget = SnapshotBudget()
    budget.add_directory(logical_path, depth)
    if directories is not None:
        directories.append(logical_path)
    root = _normal_directory(path, f"snapshot directory {logical_path!r}")
    if require_read_only:
        _require_read_only(root, f"snapshot directory {logical_path!r}")
    try:
        entries = sorted(os.scandir(root), key=lambda entry: os.fsencode(entry.name))
    except OSError as exc:
        raise TeacherLaunchError(
            f"cannot enumerate snapshot directory {logical_path!r}: {exc}"
        ) from exc
    for entry in entries:
        child_logical = f"{logical_path}/{entry.name}"
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise TeacherLaunchError(
                f"cannot inspect snapshot entry {child_logical!r}: {exc}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise TeacherLaunchError(f"snapshot contains symlink: {child_logical!r}")
        if stat.S_ISDIR(info.st_mode):
            _scan_snapshot_tree(
                Path(entry.path),
                child_logical,
                records,
                require_read_only=require_read_only,
                budget=budget,
                directories=directories,
                depth=depth + 1,
            )
        elif stat.S_ISREG(info.st_mode):
            digest, byte_len = _hash_snapshot_file(
                Path(entry.path),
                child_logical,
                require_read_only=require_read_only,
            )
            budget.add_file(child_logical, byte_len, depth + 1)
            records.append(SnapshotFile(child_logical, byte_len, digest))
        else:
            raise TeacherLaunchError(f"snapshot contains special file: {child_logical!r}")


def _read_strict_regular_file(path: Path, label: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise TeacherLaunchError(f"cannot open {label}: {exc}") from exc
    try:
        initial = os.fstat(fd)
        if not stat.S_ISREG(initial.st_mode):
            raise TeacherLaunchError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        remaining = initial.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise TeacherLaunchError(f"{label} became shorter while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise TeacherLaunchError(f"{label} became larger while being read")
        after = os.fstat(fd)
        path_after = path.stat(follow_symlinks=False)
        identity = _model_fingerprint._stat_identity(initial)
        if (
            _model_fingerprint._stat_identity(after) != identity
            or _model_fingerprint._stat_identity(path_after) != identity
        ):
            raise TeacherLaunchError(f"{label} changed while being read")
        return b"".join(chunks)
    except OSError as exc:
        raise TeacherLaunchError(f"cannot read {label}: {exc}") from exc
    finally:
        os.close(fd)


def _remove_private_child(
    parent_fd: int,
    name: str,
    *,
    expected_identity: tuple[Any, ...] | None = None,
    expected_device: int | None = None,
) -> None:
    """Remove one direct child without ever resolving through the root path."""

    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    identity = _model_fingerprint._stat_identity(info)
    if expected_identity is not None and identity != expected_identity:
        raise TeacherLaunchError(f"refusing to remove replaced snapshot entry {name!r}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        if (
            expected_device is not None
            and not stat.S_ISLNK(info.st_mode)
            and info.st_dev != expected_device
        ):
            raise TeacherLaunchError(f"refusing to cross a filesystem boundary at {name!r}")
        os.unlink(name, dir_fd=parent_fd)
        return
    if expected_device is not None and info.st_dev != expected_device:
        raise TeacherLaunchError(f"refusing to cross a filesystem boundary at {name!r}")

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    child_fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(child_fd)
        if _model_fingerprint._stat_identity(opened) != identity:
            raise TeacherLaunchError(f"snapshot entry changed while opening {name!r}")
        os.fchmod(child_fd, 0o700)
        entries = list(os.scandir(child_fd))
        for entry in entries:
            _remove_private_child(
                child_fd,
                entry.name,
                expected_device=expected_device if expected_device is not None else info.st_dev,
            )
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _validate_snapshot_child(root: Path, path: Path, prefix: str) -> None:
    absolute_root = Path(os.path.abspath(os.fspath(root)))
    absolute_path = Path(os.path.abspath(os.fspath(path)))
    if absolute_path.parent != absolute_root or not absolute_path.name.startswith(prefix):
        raise TeacherLaunchError(
            f"refusing snapshot cleanup outside {absolute_root}: {absolute_path}"
        )


def create_immutable_snapshot(
    model_source: Path,
    adapter_source: Path | None,
    snapshot_root: Path,
) -> ImmutableSnapshot:
    model = _normal_directory(model_source, "model source")
    adapter = (
        _normal_directory(adapter_source, "adapter source")
        if adapter_source is not None
        else None
    )
    _assert_no_symlink_components(model, "model source")
    if adapter is not None:
        _assert_no_symlink_components(adapter, "adapter source")
    root_candidate = Path(os.path.abspath(os.fspath(snapshot_root.expanduser())))
    for source, label in ((model, "model source"), (adapter, "adapter source")):
        if source is not None and (
            _path_is_within(root_candidate, source)
            or _path_is_within(source, root_candidate)
        ):
            raise TeacherLaunchError(
                f"snapshot root and {label} must be separate, non-nested directories: {source}"
            )
    root = _secure_snapshot_root(root_candidate)
    inventory_budget = SnapshotBudget()
    model_files, model_bytes = _inventory_source_tree(
        model, "model", inventory_budget
    )
    adapter_files, adapter_bytes = (
        _inventory_source_tree(adapter, "adapter", inventory_budget)
        if adapter is not None
        else (0, 0)
    )
    total_files = model_files + adapter_files
    total_bytes = model_bytes + adapter_bytes
    if total_files > MAX_SNAPSHOT_FILES or total_bytes > MAX_SNAPSHOT_BYTES:
        raise TeacherLaunchError("combined model and adapter snapshot exceeds its safety bounds")
    available_bytes, available_inodes, block_size = _snapshot_filesystem_capacity(root)
    entry_count = inventory_budget.files + inventory_budget.directories + 2
    metadata_bytes = (
        inventory_budget.path_bytes * 6
        + inventory_budget.files * 256
        + inventory_budget.directories * 128
    )
    allocation_floor = entry_count * max(block_size, 1)
    required_bytes = total_bytes + max(
        SNAPSHOT_HEADROOM_BYTES,
        total_bytes // 100,
        metadata_bytes,
        allocation_floor,
    )
    if available_bytes < required_bytes:
        raise TeacherLaunchError(
            "snapshot filesystem has insufficient free space for copy fallback: "
            f"requires {required_bytes} bytes, has {available_bytes}"
        )
    if available_inodes is not None and available_inodes < entry_count:
        raise TeacherLaunchError(
            "snapshot filesystem has insufficient free inodes: "
            f"requires {entry_count}, has {available_inodes}"
        )

    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root_fd = os.open(root, root_flags)
    except OSError as exc:
        raise TeacherLaunchError(f"cannot anchor snapshot root {root}: {exc}") from exc
    root_info = os.fstat(root_fd)
    root_identity = _directory_anchor_identity(root_info)
    building_name = f".building-{os.getpid()}-{secrets.token_hex(16)}"
    ready_name = f"ready-{os.getpid()}-{secrets.token_hex(16)}"
    building = root / building_name
    ready = root / ready_name
    try:
        os.mkdir(building_name, mode=0o700, dir_fd=root_fd)
        records: list[SnapshotFile] = []
        directories: list[str] = []
        copy_budget = SnapshotBudget(
            expected_files=inventory_budget.files,
            expected_directories=inventory_budget.directories,
            expected_bytes=inventory_budget.bytes,
            expected_path_bytes=inventory_budget.path_bytes,
        )
        _copy_snapshot_tree(
            model,
            building / "model",
            "model",
            records,
            directories,
            copy_budget,
        )
        if adapter is not None:
            _copy_snapshot_tree(
                adapter,
                building / "adapter",
                "adapter",
                records,
                directories,
                copy_budget,
            )
        if (
            copy_budget.files != inventory_budget.files
            or copy_budget.directories != inventory_budget.directories
            or copy_budget.bytes != inventory_budget.bytes
            or copy_budget.path_bytes != inventory_budget.path_bytes
        ):
            raise TeacherLaunchError("snapshot source changed after its capacity inventory")
        records.sort()
        directories.sort(key=lambda value: value.encode("utf-8"))
        files = tuple(records)
        directory_records = tuple(directories)
        source_records: list[SnapshotFile] = []
        source_directories: list[str] = []
        source_budget = SnapshotBudget(
            expected_files=inventory_budget.files,
            expected_directories=inventory_budget.directories,
            expected_bytes=inventory_budget.bytes,
            expected_path_bytes=inventory_budget.path_bytes,
        )
        _scan_snapshot_tree(
            model,
            "model",
            source_records,
            require_read_only=False,
            budget=source_budget,
            directories=source_directories,
        )
        if adapter is not None:
            _scan_snapshot_tree(
                adapter,
                "adapter",
                source_records,
                require_read_only=False,
                budget=source_budget,
                directories=source_directories,
            )
        source_records.sort()
        source_directories.sort(key=lambda value: value.encode("utf-8"))
        if tuple(source_records) != files or tuple(source_directories) != directory_records:
            raise TeacherLaunchError(
                "snapshot source changed across the complete multi-file copy"
            )
        manifest_payload = _snapshot_manifest_bytes(
            files,
            adapter is not None,
            directory_records,
        )
        if len(manifest_payload) > MAX_SNAPSHOT_MANIFEST_BYTES:
            raise TeacherLaunchError(
                f"snapshot manifest exceeds {MAX_SNAPSHOT_MANIFEST_BYTES} bytes"
            )
        manifest_hash = _write_snapshot_manifest(
            building / SNAPSHOT_MANIFEST,
            manifest_payload,
        )
        _freeze_snapshot_directories(building)
        candidate = ImmutableSnapshot(
            snapshot_root=root,
            path=building,
            model_path=building / "model",
            adapter_path=building / "adapter" if adapter is not None else None,
            directories=directory_records,
            files=files,
            manifest_sha256=manifest_hash,
            _root_fd=root_fd,
            _root_identity=root_identity,
            _snapshot_identity=_model_fingerprint._stat_identity(
                os.stat(building_name, dir_fd=root_fd, follow_symlinks=False)
            ),
            _snapshot_device=root_info.st_dev,
        )
        candidate.verify()
        os.rename(
            building_name,
            ready_name,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
        try:
            os.fsync(root_fd)
        except OSError as exc:
            raise TeacherLaunchError(f"cannot synchronize snapshot root {root}: {exc}") from exc
        candidate.path = ready
        candidate.model_path = ready / "model"
        candidate.adapter_path = ready / "adapter" if adapter is not None else None
        candidate._snapshot_identity = _model_fingerprint._stat_identity(
            os.stat(ready_name, dir_fd=root_fd, follow_symlinks=False)
        )
        candidate.verify()
        return candidate
    except BaseException:
        try:
            _remove_private_child(root_fd, building_name, expected_device=root_info.st_dev)
            _remove_private_child(root_fd, ready_name, expected_device=root_info.st_dev)
        finally:
            os.close(root_fd)
        raise


@dataclass
class _RuntimeContentInventory:
    files: dict[str, tuple[Path, Path]]
    directories: dict[str, Path]
    budget: RuntimeContentBudget

    @classmethod
    def empty(cls) -> "_RuntimeContentInventory":
        return cls(files={}, directories={}, budget=RuntimeContentBudget())

    def add_directory(self, label: str, path: Path, depth: int) -> None:
        resolved = _resolve_runtime_path(path, label)
        try:
            info = resolved.stat(follow_symlinks=False)
        except OSError as exc:
            raise TeacherLaunchError(
                f"cannot inspect Python runtime directory {label!r}: {exc}"
            ) from exc
        if not stat.S_ISDIR(info.st_mode):
            raise TeacherLaunchError(f"Python runtime entry is not a directory: {label!r}")
        previous = self.directories.get(label)
        if previous is not None:
            if previous != resolved:
                raise TeacherLaunchError(
                    f"Python runtime directory label resolves ambiguously: {label!r}"
                )
            return
        self.budget.add_directory(label, depth)
        self.directories[label] = resolved

    def add_file(self, label: str, path: Path, depth: int) -> None:
        absolute = Path(os.path.abspath(os.fspath(path)))
        resolved = _resolve_runtime_path(absolute, label)
        try:
            info = resolved.stat(follow_symlinks=False)
        except OSError as exc:
            raise TeacherLaunchError(
                f"cannot inspect Python runtime file {label!r}: {exc}"
            ) from exc
        if not stat.S_ISREG(info.st_mode):
            raise TeacherLaunchError(f"Python runtime entry is not a regular file: {label!r}")
        previous = self.files.get(label)
        if previous is not None:
            if previous != (absolute, resolved):
                raise TeacherLaunchError(
                    f"Python runtime file label resolves ambiguously: {label!r}"
                )
            return
        self.budget.add_file(label, info.st_size, depth)
        self.files[label] = (absolute, resolved)


def _resolve_runtime_path(path: Path, label: str) -> Path:
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TeacherLaunchError(
            f"cannot resolve Python runtime content {label!r}: {exc}"
        ) from exc


def _scan_runtime_content_tree(
    path: Path,
    label: str,
    inventory: _RuntimeContentInventory,
    *,
    depth: int = 0,
    active_directories: set[tuple[int, int]] | None = None,
) -> None:
    if active_directories is None:
        active_directories = set()
    resolved = _resolve_runtime_path(path, label)
    try:
        info = resolved.stat(follow_symlinks=False)
    except OSError as exc:
        raise TeacherLaunchError(
            f"cannot inspect Python runtime directory {label!r}: {exc}"
        ) from exc
    if not stat.S_ISDIR(info.st_mode):
        raise TeacherLaunchError(f"Python runtime tree root is not a directory: {label!r}")
    directory_identity = (info.st_dev, info.st_ino)
    if directory_identity in active_directories:
        raise TeacherLaunchError(f"Python runtime content contains a directory cycle: {label!r}")
    inventory.add_directory(label, resolved, depth)
    active_directories.add(directory_identity)
    try:
        try:
            entries = sorted(os.scandir(resolved), key=lambda entry: os.fsencode(entry.name))
        except OSError as exc:
            raise TeacherLaunchError(
                f"cannot enumerate Python runtime directory {label!r}: {exc}"
            ) from exc
        for entry in entries:
            child_label = f"{label}/{entry.name}"
            child_path = Path(entry.path)
            try:
                entry_info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise TeacherLaunchError(
                    f"cannot inspect Python runtime entry {child_label!r}: {exc}"
                ) from exc
            if stat.S_ISLNK(entry_info.st_mode):
                target = _resolve_runtime_path(child_path, child_label)
                try:
                    target_info = target.stat(follow_symlinks=False)
                except OSError as exc:
                    raise TeacherLaunchError(
                        f"cannot inspect Python runtime symlink target {child_label!r}: {exc}"
                    ) from exc
                if stat.S_ISDIR(target_info.st_mode):
                    _scan_runtime_content_tree(
                        child_path,
                        child_label,
                        inventory,
                        depth=depth + 1,
                        active_directories=active_directories,
                    )
                elif stat.S_ISREG(target_info.st_mode):
                    inventory.add_file(child_label, child_path, depth + 1)
                else:
                    raise TeacherLaunchError(
                        f"Python runtime symlink targets a special file: {child_label!r}"
                    )
            elif stat.S_ISDIR(entry_info.st_mode):
                _scan_runtime_content_tree(
                    child_path,
                    child_label,
                    inventory,
                    depth=depth + 1,
                    active_directories=active_directories,
                )
            elif stat.S_ISREG(entry_info.st_mode):
                inventory.add_file(child_label, child_path, depth + 1)
            else:
                raise TeacherLaunchError(
                    f"Python runtime content contains a special file: {child_label!r}"
                )
    finally:
        active_directories.remove(directory_identity)


def _package_import_root(package: str) -> tuple[Path | None, Path | None]:
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, AttributeError, ValueError) as exc:
        raise TeacherLaunchError(f"cannot resolve Python package {package!r}: {exc}") from exc
    if spec is None:
        raise TeacherLaunchError(f"required Python package {package!r} cannot be imported")
    locations = tuple(spec.submodule_search_locations or ())
    if len(locations) > 1:
        raise TeacherLaunchError(
            f"Python package {package!r} resolves from multiple namespace roots; "
            "the immutable launcher requires one unambiguous import tree"
        )
    root = Path(locations[0]) if locations else None
    origin_value = spec.origin
    origin = None
    if isinstance(origin_value, str) and origin_value not in {"built-in", "frozen"}:
        origin = Path(origin_value)
    if root is None and origin is None:
        raise TeacherLaunchError(
            f"Python package {package!r} has neither an import tree nor a file origin"
        )
    return root, origin


def _distribution_metadata_path(distribution: Any) -> Path | None:
    path = getattr(distribution, "_path", None)
    return Path(path) if path is not None else None


def _collect_runtime_content(
    python_executable: Path,
    package_names: Sequence[str],
) -> _RuntimeContentInventory:
    inventory = _RuntimeContentInventory.empty()
    inventory.add_file("python/executable", python_executable, 0)

    for package in package_names:
        if not package or any(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) is None
            for part in package.split(".")
        ):
            raise TeacherLaunchError(f"invalid Python runtime package name: {package!r}")
        import_root, import_origin = _package_import_root(package)
        if import_root is not None:
            _scan_runtime_content_tree(
                import_root,
                f"package/{package}/import",
                inventory,
            )
        if import_origin is not None:
            resolved_origin = _resolve_runtime_path(
                import_origin, f"package/{package}/import-origin"
            )
            origin_covered = import_root is not None and _path_is_within(
                resolved_origin,
                _resolve_runtime_path(import_root, f"package/{package}/import"),
            )
            if not origin_covered:
                inventory.add_file(
                    f"package/{package}/import-origin/{resolved_origin.name}",
                    import_origin,
                    1,
                )

        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise TeacherLaunchError(
                f"installed distribution metadata is missing for Python package {package!r}"
            ) from exc
        metadata_path = _distribution_metadata_path(distribution)
        if metadata_path is not None:
            resolved_metadata = _resolve_runtime_path(
                metadata_path, f"package/{package}/metadata"
            )
            if resolved_metadata.is_dir():
                _scan_runtime_content_tree(
                    metadata_path,
                    f"package/{package}/metadata",
                    inventory,
                )
            elif resolved_metadata.is_file():
                inventory.add_file(
                    f"package/{package}/metadata/{resolved_metadata.name}",
                    metadata_path,
                    1,
                )
            else:
                raise TeacherLaunchError(
                    f"distribution metadata for {package!r} is not a regular file or directory"
                )

        distribution_files = distribution.files
        if distribution_files is None:
            # Legacy editable installs often omit RECORD entries. The import
            # tree above is authoritative in that case; metadata is still
            # included when importlib exposes its concrete path.
            continue
        for item in sorted(distribution_files, key=lambda value: os.fsencode(os.fspath(value))):
            raw_label = os.fspath(item).replace(os.sep, "/")
            label = f"package/{package}/distribution/{raw_label}"
            try:
                located = Path(distribution.locate_file(item))
            except (OSError, TypeError, ValueError) as exc:
                raise TeacherLaunchError(
                    f"cannot locate installed distribution file {label!r}: {exc}"
                ) from exc
            resolved = _resolve_runtime_path(located, label)
            if resolved.is_dir():
                _scan_runtime_content_tree(located, label, inventory)
            elif resolved.is_file():
                inventory.add_file(label, located, 1)
            else:
                raise TeacherLaunchError(
                    f"installed distribution entry is not a regular file or directory: {label!r}"
                )
    return inventory


def _hash_runtime_content_file(label: str, original: Path, resolved: Path) -> RuntimeContentRecord:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(resolved, flags)
    except OSError as exc:
        raise TeacherLaunchError(f"cannot open Python runtime file {label!r}: {exc}") from exc
    try:
        initial = os.fstat(fd)
        if not stat.S_ISREG(initial.st_mode):
            raise TeacherLaunchError(f"Python runtime entry is not a regular file: {label!r}")
        if initial.st_size > MAX_RUNTIME_CONTENT_BYTES:
            raise TeacherLaunchError(
                f"Python runtime file exceeds the logical-size safety limit: {label!r}"
            )
        digest, byte_len = _hash_fd(fd, initial.st_size)
        after = os.fstat(fd)
        path_after = resolved.stat(follow_symlinks=False)
        identity = _model_fingerprint._stat_identity(initial)
        if (
            _model_fingerprint._stat_identity(after) != identity
            or _model_fingerprint._stat_identity(path_after) != identity
        ):
            raise TeacherLaunchError(f"Python runtime file changed while hashed: {label!r}")
        if _resolve_runtime_path(original, label) != resolved:
            raise TeacherLaunchError(f"Python runtime path changed while hashed: {label!r}")
        return RuntimeContentRecord(label=label, byte_len=byte_len, sha256=digest)
    except OSError as exc:
        raise TeacherLaunchError(f"cannot hash Python runtime file {label!r}: {exc}") from exc
    finally:
        os.close(fd)


def capture_runtime_content(
    *,
    python_executable: Path | None = None,
    package_names: Sequence[str] = RUNTIME_PACKAGES,
) -> RuntimeContentSnapshot:
    executable = _resolve_runtime_path(
        Path(sys.executable) if python_executable is None else python_executable,
        "python/executable",
    )
    packages = tuple(package_names)
    inventory = _collect_runtime_content(executable, packages)
    records: list[RuntimeContentRecord] = []
    hashed_paths: dict[Path, tuple[str, int]] = {}
    for label, (original, resolved) in sorted(inventory.files.items()):
        cached = hashed_paths.get(resolved)
        if cached is None:
            record = _hash_runtime_content_file(label, original, resolved)
            cached = (record.sha256, record.byte_len)
            hashed_paths[resolved] = cached
        else:
            if _resolve_runtime_path(original, label) != resolved:
                raise TeacherLaunchError(f"Python runtime path changed while hashed: {label!r}")
            record = RuntimeContentRecord(label=label, byte_len=cached[1], sha256=cached[0])
        records.append(record)

    observed_after = _collect_runtime_content(executable, packages)
    expected_files = {
        label: resolved for label, (_original, resolved) in inventory.files.items()
    }
    observed_files = {
        label: resolved for label, (_original, resolved) in observed_after.files.items()
    }
    if expected_files != observed_files or inventory.directories != observed_after.directories:
        raise TeacherLaunchError("Python runtime content changed across its complete hash")
    if (
        inventory.budget.files != observed_after.budget.files
        or inventory.budget.directories != observed_after.budget.directories
        or inventory.budget.logical_bytes != observed_after.budget.logical_bytes
        or inventory.budget.path_bytes != observed_after.budget.path_bytes
    ):
        raise TeacherLaunchError("Python runtime content bounds changed across its complete hash")

    digest = hashlib.sha256()
    digest.update(RUNTIME_CONTENT_DOMAIN)
    _feed_bytes(digest, RUNTIME_CONTENT_SCHEMA.encode("ascii"))
    directory_labels = sorted(inventory.directories, key=lambda value: value.encode("utf-8"))
    digest.update(struct.pack("<Q", len(directory_labels)))
    for label in directory_labels:
        _feed_bytes(digest, label.encode("utf-8"))
    records.sort()
    digest.update(struct.pack("<Q", len(records)))
    for record in records:
        _feed_bytes(digest, record.label.encode("utf-8"))
        digest.update(struct.pack("<Q", record.byte_len))
        _feed_hash(digest, record.sha256, record.label)
    return RuntimeContentSnapshot(
        sha256=digest.hexdigest(),
        python_executable=executable,
        file_count=inventory.budget.files,
        directory_count=inventory.budget.directories,
        logical_bytes=inventory.budget.logical_bytes,
    )


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


def _model_vocab_size(model_config: Mapping[str, Any]) -> int:
    """Resolve the model's embedding/logit width from common HF config layouts."""

    candidates: list[tuple[str, Any]] = []
    if "vocab_size" in model_config and model_config["vocab_size"] is not None:
        candidates.append(("config.json vocab_size", model_config["vocab_size"]))
    text_config = model_config.get("text_config")
    if isinstance(text_config, Mapping) and text_config.get("vocab_size") is not None:
        candidates.append(
            ("config.json text_config.vocab_size", text_config["vocab_size"])
        )
    if not candidates:
        raise TeacherLaunchError(
            "config.json must declare vocab_size or text_config.vocab_size"
        )

    for label, value in candidates:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TeacherLaunchError(f"{label} must be a positive integer")
    values = {value for _, value in candidates}
    if len(values) != 1:
        rendered = ", ".join(f"{label}={value}" for label, value in candidates)
        raise TeacherLaunchError(f"config.json vocabulary sizes disagree: {rendered}")
    return candidates[0][1]


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
        raise TeacherLaunchError(
            "immutable custom fingerprints require "
            f"vLLM {MIN_VLLM_VERSION_TEXT} or newer"
        )
    return value


def _installed_vllm_version() -> str:
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TeacherLaunchError("vLLM is not installed in this Python environment") from exc
    _validate_implementation(f"vllm:{version}")
    return version


def _bounded_runtime_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TeacherLaunchError(f"{field} must be a non-empty string")
    try:
        payload = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TeacherLaunchError(f"{field} is not valid UTF-8") from exc
    if len(payload) > MAX_RUNTIME_VALUE_BYTES:
        raise TeacherLaunchError(
            f"{field} exceeds the {MAX_RUNTIME_VALUE_BYTES}-byte runtime identity limit"
        )
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise TeacherLaunchError(f"{field} must not contain control characters")
    return value


def _validate_runtime_versions(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(RUNTIME_VERSION_FIELDS):
        raise TeacherLaunchError(
            "runtime_versions must contain exactly " + ", ".join(RUNTIME_VERSION_FIELDS)
        )
    result = {
        field: _bounded_runtime_string(value.get(field), f"runtime_versions.{field}")
        for field in RUNTIME_VERSION_FIELDS
    }
    _validate_implementation(f"vllm:{result['vllm']}")
    return result


def installed_runtime_versions(vllm_version: str | None = None) -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    for package in ("vllm", "torch", "transformers", "tokenizers"):
        if package == "vllm" and vllm_version is not None:
            version = vllm_version
        else:
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as exc:
                raise TeacherLaunchError(
                    f"{package} is not installed in this Python environment"
                ) from exc
        versions[package] = version
    return _validate_runtime_versions(versions)


def runtime_contract_probe_payload() -> dict[str, Any]:
    content = capture_runtime_content()
    return {
        "schema": RUNTIME_CONTENT_SCHEMA,
        "runtime_versions": installed_runtime_versions(),
        "runtime_content_sha256": content.sha256,
        "file_count": content.file_count,
        "directory_count": content.directory_count,
        "logical_bytes": content.logical_bytes,
    }


def _validate_runtime_probe_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(RUNTIME_PROBE_FIELDS):
        raise TeacherLaunchError(
            "child runtime probe must contain exactly " + ", ".join(RUNTIME_PROBE_FIELDS)
        )
    if value.get("schema") != RUNTIME_CONTENT_SCHEMA:
        raise TeacherLaunchError(
            f"child runtime probe schema must be {RUNTIME_CONTENT_SCHEMA!r}"
        )
    counts: dict[str, int] = {}
    for field, maximum in (
        ("file_count", MAX_RUNTIME_CONTENT_FILES),
        ("directory_count", MAX_RUNTIME_CONTENT_DIRECTORIES),
        ("logical_bytes", MAX_RUNTIME_CONTENT_BYTES),
    ):
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw <= maximum:
            raise TeacherLaunchError(f"child runtime probe {field} is outside its safety bound")
        counts[field] = raw
    return {
        "schema": RUNTIME_CONTENT_SCHEMA,
        "runtime_versions": _validate_runtime_versions(value.get("runtime_versions")),
        "runtime_content_sha256": _raw_sha256(
            value.get("runtime_content_sha256"), "runtime_content_sha256"
        ),
        **counts,
    }


def verify_child_runtime_contract(
    expected_versions: Mapping[str, str],
    expected_content_sha256: str,
    environment: Mapping[str, str],
) -> None:
    launcher_path = os.fspath(Path(__file__).resolve())
    probe = (
        "import importlib.util,json,sys;"
        f"p={launcher_path!r};"
        "s=importlib.util.spec_from_file_location('_kiln_vllm_runtime_probe',p);"
        "m=importlib.util.module_from_spec(s);"
        "sys.modules[s.name]=m;"
        "s.loader.exec_module(m);"
        "print(json.dumps(m.runtime_contract_probe_payload(),separators=(',',':')))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=os.path.abspath(os.sep),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
            timeout=RUNTIME_CONTENT_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TeacherLaunchError(
            f"failed to revalidate the exact child Python runtime content: {exc}"
        ) from exc
    if completed.returncode != 0 or len(completed.stdout) > MAX_INPUT_MANIFEST_BYTES:
        detail = completed.stderr.decode("utf-8", errors="replace")[:512].strip()
        raise TeacherLaunchError(
            "child Python cannot revalidate its required runtime content"
            + (f": {detail}" if detail else "")
        )
    observed = _validate_runtime_probe_payload(
        _strict_json_object(completed.stdout.strip(), "child runtime content probe")
    )
    if observed["runtime_versions"] != _validate_runtime_versions(expected_versions):
        raise TeacherLaunchError(
            "child Python runtime versions differ from the fingerprinting process"
        )
    expected_digest = _raw_sha256(expected_content_sha256, "runtime_content_sha256")
    if observed["runtime_content_sha256"] != expected_digest:
        raise TeacherLaunchError(
            "child Python executable or package content changed after identity construction"
        )


def _validate_accelerator(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(ACCELERATOR_FIELDS):
        raise TeacherLaunchError(
            "accelerator must contain exactly type, driver, and devices"
        )
    accelerator_type = value.get("type")
    if accelerator_type not in ACCELERATOR_TYPES:
        raise TeacherLaunchError(
            "accelerator.type must be one of " + ", ".join(sorted(ACCELERATOR_TYPES))
        )
    driver = _bounded_runtime_string(value.get("driver"), "accelerator.driver")
    devices = value.get("devices")
    if (
        not isinstance(devices, list)
        or not devices
        or len(devices) > MAX_ACCELERATOR_DEVICES
    ):
        raise TeacherLaunchError(
            f"accelerator.devices must contain 1..={MAX_ACCELERATOR_DEVICES} entries"
        )
    normalized_devices: list[dict[str, Any]] = []
    for expected_index, device in enumerate(devices):
        if not isinstance(device, Mapping) or set(device) != set(ACCELERATOR_DEVICE_FIELDS):
            raise TeacherLaunchError(
                "each accelerator device must contain exactly index, name, architecture, "
                "and total_memory_bytes"
            )
        index = device.get("index")
        memory = device.get("total_memory_bytes")
        if isinstance(index, bool) or not isinstance(index, int) or index != expected_index:
            raise TeacherLaunchError("accelerator device indexes must be contiguous from zero")
        if isinstance(memory, bool) or not isinstance(memory, int) or memory < 0:
            raise TeacherLaunchError(
                "accelerator total_memory_bytes must be a non-negative integer"
            )
        normalized_devices.append(
            {
                "index": index,
                "name": _bounded_runtime_string(
                    device.get("name"), f"accelerator.devices[{index}].name"
                ),
                "architecture": _bounded_runtime_string(
                    device.get("architecture"),
                    f"accelerator.devices[{index}].architecture",
                ),
                "total_memory_bytes": memory,
            }
        )
    return {
        "type": accelerator_type,
        "driver": driver,
        "devices": normalized_devices,
    }


def _read_bounded_text(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            payload = handle.read(MAX_RUNTIME_VALUE_BYTES + 1)
    except OSError:
        return None
    if not payload or len(payload) > MAX_RUNTIME_VALUE_BYTES:
        return None
    value = payload.decode("utf-8", errors="replace").strip()
    return value or None


def _command_output(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or len(completed.stdout) > MAX_RUNTIME_VALUE_BYTES:
        return None
    value = completed.stdout.decode("utf-8", errors="replace").strip()
    return value or None


def _gpu_driver_identity(torch_module: Any, accelerator_type: str) -> str:
    runtime_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    if accelerator_type == "rocm":
        runtime_version = getattr(getattr(torch_module, "version", None), "hip", None)
        kernel_driver = _read_bounded_text(Path("/sys/module/amdgpu/version"))
        return (
            f"amdgpu:{kernel_driver or 'kernel-' + platform.release()};"
            f"hip-runtime:{runtime_version or 'unknown'}"
        )

    driver = _command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if driver is not None:
        driver = ",".join(sorted(set(driver.splitlines())))
    else:
        proc_driver = _read_bounded_text(Path("/proc/driver/nvidia/version"))
        driver = proc_driver or f"kernel-{platform.release()}"
    return f"nvidia:{driver};cuda-runtime:{runtime_version or 'unknown'}"


def _device_properties(accelerator: Any, index: int, accelerator_type: str) -> dict[str, Any]:
    try:
        properties = accelerator.get_device_properties(index)
    except Exception as exc:
        raise TeacherLaunchError(
            f"cannot inspect {accelerator_type} device {index}: {exc}"
        ) from exc
    name = getattr(properties, "name", None)
    if not name:
        try:
            name = accelerator.get_device_name(index)
        except Exception:
            name = f"{accelerator_type}-{index}"
    if accelerator_type == "rocm":
        architecture = getattr(properties, "gcnArchName", None)
    else:
        major = getattr(properties, "major", None)
        minor = getattr(properties, "minor", None)
        architecture = (
            f"sm_{major}{minor}"
            if isinstance(major, int) and isinstance(minor, int)
            else None
        )
    if not architecture:
        try:
            capability = accelerator.get_device_capability(index)
            architecture = f"{accelerator_type}_{capability[0]}{capability[1]}"
        except Exception:
            architecture = platform.machine() or "unknown"
    memory = getattr(properties, "total_memory", 0)
    return {
        "index": index,
        "name": str(name),
        "architecture": str(architecture),
        "total_memory_bytes": int(memory),
    }


def probe_accelerator() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise TeacherLaunchError("torch is required to identify the inference accelerator") from exc

    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        accelerator_type = (
            "rocm"
            if getattr(getattr(torch, "version", None), "hip", None)
            else "cuda"
        )
        try:
            count = int(cuda.device_count())
        except Exception as exc:
            raise TeacherLaunchError(f"cannot enumerate {accelerator_type} devices: {exc}") from exc
        if count <= 0:
            raise TeacherLaunchError(
                f"torch reports {accelerator_type} available but exposes no devices"
            )
        return _validate_accelerator(
            {
                "type": accelerator_type,
                "driver": _gpu_driver_identity(torch, accelerator_type),
                "devices": [
                    _device_properties(cuda, index, accelerator_type)
                    for index in range(count)
                ],
            }
        )

    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        count = int(xpu.device_count())
        return _validate_accelerator(
            {
                "type": "xpu",
                "driver": f"level-zero:unknown;kernel:{platform.release()}",
                "devices": [
                    _device_properties(xpu, index, "xpu") for index in range(count)
                ],
            }
        )

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        name = _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        return _validate_accelerator(
            {
                "type": "metal",
                "driver": f"macos:{platform.mac_ver()[0] or platform.release()}",
                "devices": [
                    {
                        "index": 0,
                        "name": name or platform.processor() or "Apple GPU",
                        "architecture": platform.machine() or "unknown",
                        "total_memory_bytes": 0,
                    }
                ],
            }
        )

    cpu_name = platform.processor() or platform.machine() or "unknown CPU"
    cpuinfo = _read_bounded_text(Path("/proc/cpuinfo"))
    if cpuinfo:
        for line in cpuinfo.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "Hardware"} and value.strip():
                cpu_name = value.strip()
                break
    return _validate_accelerator(
        {
            "type": "cpu",
            "driver": f"{platform.system()}:{platform.release()}",
            "devices": [
                {
                    "index": 0,
                    "name": cpu_name,
                    "architecture": platform.machine() or "unknown",
                    "total_memory_bytes": 0,
                }
            ],
        }
    )


def _deterministic_policy(
    extra_args: Sequence[str], environment: Mapping[str, str]
) -> dict[str, Any]:
    seed: str | None = None
    enforce_eager = False
    for raw in extra_args:
        option, separator, value = raw.partition("=")
        if option == "--seed" and separator:
            seed = value
        elif option == "--enforce-eager":
            enforce_eager = True
    return {
        "seed": seed,
        "enforce_eager": enforce_eager,
        "environment": {key: environment.get(key) for key in DETERMINISM_ENV_KEYS},
    }


def _unbound_environment_key(key: str) -> bool:
    if key in FORBIDDEN_FILE_ENV_KEYS:
        return True
    if key in SAFE_ENV_NAME_EXCEPTIONS:
        return False
    if not (key.startswith(INFERENCE_ENV_PREFIXES) or key in INFERENCE_ENV_KEYS):
        return False
    return bool(set(key.split("_")) & UNBOUND_ENV_NAME_PARTS)


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
        option_parts = set(option[2:].split("-"))
        if (
            option not in TRANSPORT_OPTIONS
            and not option.startswith(TRANSPORT_PREFIXES)
            and option_parts & PATH_LIKE_INFERENCE_OPTION_PARTS
        ):
            raise TeacherLaunchError(
                f"unbound file or path option is forbidden by the immutable launcher: {option}"
            )
        if (
            option not in TRANSPORT_OPTIONS
            and not option.startswith(TRANSPORT_PREFIXES)
            and option not in VALUELESS_OPTIONS
            and option not in VETTED_INFERENCE_OPTIONS
        ):
            raise TeacherLaunchError(
                f"vLLM inference option has not been identity-reviewed: {option}"
            )
        if not separator and option not in VALUELESS_OPTIONS:
            raise TeacherLaunchError(f"vLLM option must use --key=value form: {option}")
        if separator and option in VALUELESS_OPTIONS:
            raise TeacherLaunchError(f"vLLM valueless option must not use =value: {option}")
        if separator and not value:
            raise TeacherLaunchError(f"vLLM option value must not be empty: {option}")
        if option == "--attention-backend" and value not in REVIEWED_ATTENTION_BACKENDS:
            reviewed = ", ".join(sorted(REVIEWED_ATTENTION_BACKENDS))
            raise TeacherLaunchError(
                f"unsupported attention backend {value!r}; reviewed values: {reviewed}"
            )
        result.append(raw)
    return result


def inference_config_fingerprint(
    *,
    snapshot_content_sha256: str,
    model_config_sha256: str,
    max_top_k: int,
    max_model_len: int,
    max_prompt_logprob_candidates: int,
    adapter_enabled: bool,
    adapter_max_rank: int | None,
    extra_args: Sequence[str],
    environment: Mapping[str, str],
    runtime_versions: Mapping[str, str],
    runtime_content_sha256: str,
    accelerator: Mapping[str, Any],
) -> str:
    for key, raw_value in environment.items():
        if raw_value.strip() and _unbound_environment_key(key):
            raise TeacherLaunchError(
                f"{key} is forbidden because file content cannot be represented by path alone"
            )
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
        if (key.startswith(INFERENCE_ENV_PREFIXES) or key in INFERENCE_ENV_KEYS)
        and key not in {"VLLM_ALLOW_RUNTIME_LORA_UPDATING"}
    }
    runtime_value = _validate_runtime_versions(runtime_versions)
    accelerator_value = _validate_accelerator(accelerator)
    value = {
        "schema": INFERENCE_CONFIG_SCHEMA,
        "snapshot_content_sha256": _raw_sha256(
            snapshot_content_sha256, "snapshot_content_sha256"
        ),
        "model_config_sha256": _raw_sha256(
            model_config_sha256, "model_config_sha256"
        ),
        "max_top_k": max_top_k,
        "max_model_len": max_model_len,
        "max_prompt_logprob_candidates": max_prompt_logprob_candidates,
        "logprobs_mode": LOGPROBS_MODE,
        "generation_config": "vllm",
        "load_format": "safetensors",
        "adapter_enabled": adapter_enabled,
        "adapter_max_rank": adapter_max_rank,
        "runtime_lora_updates": False,
        "vllm_args": inference_args,
        "environment": inference_environment,
        "deterministic_policy": _deterministic_policy(validated, environment),
        "runtime_versions": runtime_value,
        "runtime_content_sha256": _raw_sha256(
            runtime_content_sha256, "runtime_content_sha256"
        ),
        "accelerator": accelerator_value,
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
    max_prompt_logprob_candidates: int,
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
    one_row_max = min(max_top_k + 1, vocab_size)
    theoretical_max = max_model_len * one_row_max
    if (
        isinstance(max_prompt_logprob_candidates, bool)
        or not isinstance(max_prompt_logprob_candidates, int)
        or not one_row_max
        <= max_prompt_logprob_candidates
        <= min(MAX_PROMPT_LOGPROB_CANDIDATES, theoretical_max)
    ):
        raise TeacherLaunchError(
            "max_prompt_logprob_candidates must fit one maximum-K row and not exceed "
            f"min({MAX_PROMPT_LOGPROB_CANDIDATES}, max_model_len * row_width)"
        )
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
        "max_prompt_logprob_candidates": max_prompt_logprob_candidates,
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
        max_prompt_logprob_candidates=identity["max_prompt_logprob_candidates"],
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
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(absolute, flags)
    except OSError as exc:
        raise TeacherLaunchError(f"cannot open identity input {absolute}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise TeacherLaunchError("identity input is not a regular file")
        if before.st_size > MAX_INPUT_MANIFEST_BYTES:
            raise TeacherLaunchError(
                f"identity input exceeds {MAX_INPUT_MANIFEST_BYTES} bytes"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise TeacherLaunchError("identity input became shorter while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise TeacherLaunchError("identity input became larger while being read")
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
        raise TeacherLaunchError(
            f"identity input has missing keys {missing} and extra keys {extra}"
        )
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
    implementation = _validate_implementation(value.get("implementation"))
    runtime_versions = _validate_runtime_versions(value.get("runtime_versions"))
    if implementation != f"vllm:{runtime_versions['vllm']}":
        raise TeacherLaunchError(
            "identity input implementation must match runtime_versions.vllm"
        )
    return {
        "base_model_sha256": _raw_sha256(
            value.get("base_model_sha256"), "base_model_sha256"
        ),
        "snapshot_content_sha256": _raw_sha256(
            value.get("snapshot_content_sha256"), "snapshot_content_sha256"
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
        "implementation": implementation,
        "runtime_versions": runtime_versions,
        "runtime_content_sha256": _raw_sha256(
            value.get("runtime_content_sha256"), "runtime_content_sha256"
        ),
        "accelerator": _validate_accelerator(value.get("accelerator")),
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
        base_hash = decode_system_fingerprint(system_fingerprint)["base_model_sha256"]
        base_name = f"kiln-base-{base_hash[:16]}"
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
        "--load-format=safetensors",
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
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("VLLM_LORA_RESOLVER_CACHE_DIR", None)
    for key in FORBIDDEN_FILE_ENV_KEYS:
        environment.pop(key, None)
    return environment


def validate_launch_environment(environment: Mapping[str, str]) -> None:
    for key, raw_value in environment.items():
        if raw_value.strip() and _unbound_environment_key(key):
            raise TeacherLaunchError(
                f"{key} is forbidden because its referenced code or file bytes "
                "are not identity-bound"
            )
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


def _process_group_exists(process_group: int) -> bool:
    if not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_process_group(process_group: int, signum: int) -> None:
    try:
        os.killpg(process_group, signum)
    except ProcessLookupError:
        pass


def _wait_process_group_gone(process_group: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _process_group_exists(process_group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _drain_process_group(process_group: int) -> None:
    """Terminate descendants left behind after the supervised leader exits."""

    if not _process_group_exists(process_group):
        return
    _signal_process_group(process_group, signal.SIGTERM)
    if _wait_process_group_gone(process_group, PROCESS_GROUP_TERM_SECONDS):
        return
    _signal_process_group(process_group, signal.SIGKILL)
    if not _wait_process_group_gone(process_group, PROCESS_GROUP_KILL_SECONDS):
        raise TeacherLaunchError(
            f"vLLM process group {process_group} did not exit after SIGKILL"
        )


def _proc_process_identity(pid: int) -> tuple[int, int] | None:
    """Return (process group, start ticks) without trusting a reused PID."""

    try:
        payload = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    except (FileNotFoundError, PermissionError, OSError, UnicodeError):
        return None
    closing = payload.rfind(")")
    if closing < 0:
        return None
    fields = payload[closing + 2 :].split()
    if len(fields) < 20:
        return None
    try:
        return int(fields[2]), int(fields[19])
    except ValueError:
        return None


def _inherited_process_group_members(process_group: int) -> dict[int, int]:
    members: dict[int, int] = {}
    try:
        entries = os.scandir("/proc")
    except OSError as exc:
        raise TeacherLaunchError(
            f"cannot enumerate inherited process group {process_group}: {exc}"
        ) from exc
    with entries:
        for entry in entries:
            if not entry.name.isascii() or not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == os.getpid():
                continue
            identity = _proc_process_identity(pid)
            if identity is not None and identity[0] == process_group:
                members[pid] = identity[1]
    return members


def _signal_inherited_group_members(process_group: int, signum: int) -> None:
    for pid, start_ticks in _inherited_process_group_members(process_group).items():
        if _proc_process_identity(pid) != (process_group, start_ticks):
            continue
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def _wait_inherited_group_empty(process_group: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _inherited_process_group_members(process_group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _drain_inherited_process_group(process_group: int) -> None:
    """Terminate every peer while leaving the external supervisor alive."""

    if not _inherited_process_group_members(process_group):
        return
    _signal_inherited_group_members(process_group, signal.SIGTERM)
    if _wait_inherited_group_empty(process_group, PROCESS_GROUP_TERM_SECONDS):
        return
    _signal_inherited_group_members(process_group, signal.SIGKILL)
    if not _wait_inherited_group_empty(process_group, PROCESS_GROUP_KILL_SECONDS):
        raise TeacherLaunchError(
            f"inherited vLLM process group {process_group} retained peers after SIGKILL"
        )


def _require_isolated_inherited_process_group() -> int:
    if not hasattr(os, "getpgrp") or not Path("/proc/self/stat").is_file():
        raise TeacherLaunchError(
            "inherited process-group mode requires Linux /proc and POSIX process groups"
        )
    process_group = os.getpgrp()
    if process_group != os.getpid():
        raise TeacherLaunchError(
            "inherited process-group mode requires the launcher to lead an isolated group"
        )
    return process_group


def _terminate_supervised_child(child: Any, process_group_mode: str) -> None:
    if process_group_mode == PROCESS_GROUP_MODE_INHERITED:
        process_group = os.getpgrp()
        _signal_inherited_group_members(process_group, signal.SIGTERM)
    elif hasattr(os, "killpg"):
        _signal_process_group(child.pid, signal.SIGTERM)
    elif child.poll() is None:
        child.terminate()
    try:
        child.wait(timeout=PROCESS_GROUP_TERM_SECONDS)
    except subprocess.TimeoutExpired:
        if process_group_mode == PROCESS_GROUP_MODE_INHERITED:
            _signal_inherited_group_members(os.getpgrp(), signal.SIGKILL)
        elif hasattr(os, "killpg"):
            _signal_process_group(child.pid, signal.SIGKILL)
        else:
            child.kill()
        child.wait()
    if process_group_mode == PROCESS_GROUP_MODE_INHERITED:
        _drain_inherited_process_group(os.getpgrp())
    elif hasattr(os, "killpg"):
        _drain_process_group(child.pid)


def run_vllm_child(
    command: Sequence[str],
    environment: Mapping[str, str],
    process_group_mode: str = PROCESS_GROUP_MODE_DETACHED,
) -> int:
    """Run vLLM without a shell while retaining snapshot cleanup ownership."""

    if process_group_mode not in PROCESS_GROUP_MODES:
        raise TeacherLaunchError(
            f"unsupported process-group mode: {process_group_mode!r}"
        )
    inherited_process_group = (
        _require_isolated_inherited_process_group()
        if process_group_mode == PROCESS_GROUP_MODE_INHERITED
        else None
    )
    child: Any | None = None
    pending_signals: list[int] = []
    previous_handlers: dict[int, Any] = {}

    def forward_signal(signum: int, _frame: Any) -> None:
        if child is None:
            pending_signals.append(signum)
            return
        if process_group_mode == PROCESS_GROUP_MODE_INHERITED:
            if child.poll() is None:
                try:
                    child.send_signal(signum)
                except ProcessLookupError:
                    pass
        elif hasattr(os, "killpg"):
            _signal_process_group(child.pid, signum)
        elif child.poll() is None:
            child.send_signal(signum)

    # Install forwarding before spawn so no signal can strand a newly detached
    # child between Popen and handler installation.
    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)
        except (ValueError, OSError) as exc:
            previous_handlers.pop(signum, None)
            for installed_signum, handler in previous_handlers.items():
                try:
                    signal.signal(installed_signum, handler)
                except (ValueError, OSError):
                    pass
            raise TeacherLaunchError(
                "vLLM supervision requires signal handlers on the main thread"
            ) from exc

    try:
        try:
            child = subprocess.Popen(
                list(command),
                env=dict(environment),
                cwd=os.path.abspath(os.sep),
                shell=False,
                start_new_session=process_group_mode == PROCESS_GROUP_MODE_DETACHED,
            )
        except OSError as exc:
            raise TeacherLaunchError(f"failed to start vLLM: {exc}") from exc
        for signum in pending_signals:
            forward_signal(signum, None)
        return_code = child.wait()
        if inherited_process_group is not None:
            _drain_inherited_process_group(inherited_process_group)
        elif hasattr(os, "killpg"):
            _drain_process_group(child.pid)
        return return_code if return_code >= 0 else 128 + (-return_code)
    except BaseException:
        if child is not None:
            _terminate_supervised_child(child, process_group_mode)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="local base model directory")
    parser.add_argument(
        "--served-model-id",
        required=True,
        help="identity-bound request and response model ID",
    )
    parser.add_argument("--adapter-path", type=Path, help="one immutable static LoRA adapter")
    parser.add_argument(
        "--process-group-mode",
        choices=PROCESS_GROUP_MODES,
        default=PROCESS_GROUP_MODE_DETACHED,
        help=(
            "detach vLLM into a launcher-owned group, or inherit an already isolated "
            "external supervisor group"
        ),
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=default_snapshot_root(),
        help=(
            "private staging directory (default: KILN_VLLM_SNAPSHOT_ROOT or "
            "~/.cache/kiln/teacher-snapshots)"
        ),
    )
    parser.add_argument("--max-top-k", required=True, type=int, help="maximum prompt_logprobs K")
    parser.add_argument("--max-model-len", required=True, type=int, help="maximum token context")
    parser.add_argument(
        "--max-prompt-logprob-candidates",
        type=int,
        help="combined response candidate cap (default: min(1,000,000, context * row width))",
    )
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


def _validate_requested_limits(args: argparse.Namespace) -> None:
    if (
        isinstance(args.max_top_k, bool)
        or not isinstance(args.max_top_k, int)
        or not 1 <= args.max_top_k <= MAX_TOP_K
    ):
        raise TeacherLaunchError(f"--max-top-k must be in 1..={MAX_TOP_K}")
    if (
        isinstance(args.max_model_len, bool)
        or not isinstance(args.max_model_len, int)
        or not 1 <= args.max_model_len <= MAX_MODEL_LEN
    ):
        raise TeacherLaunchError(f"--max-model-len must be in 1..={MAX_MODEL_LEN}")
    candidates = args.max_prompt_logprob_candidates
    if candidates is not None and (
        isinstance(candidates, bool)
        or not isinstance(candidates, int)
        or not 1 <= candidates <= MAX_PROMPT_LOGPROB_CANDIDATES
        or candidates > args.max_model_len * (args.max_top_k + 1)
    ):
        raise TeacherLaunchError(
            "--max-prompt-logprob-candidates is outside the requested context/K bounds"
        )


def _identity_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if args.identity_input is not None:
        if not (args.manifest_only or args.dry_run):
            raise TeacherLaunchError("--identity-input is forbidden for a real launch")
        return load_identity_input(args.identity_input)
    if args.model_path is None:
        raise TeacherLaunchError("--model-path is required without --identity-input")
    model_path = _normal_directory(args.model_path, "model path")
    snapshot_hash = getattr(args, "snapshot_content_sha256", None)
    if snapshot_hash is None:
        snapshot_hash = snapshot_content_fingerprint(model_path, args.adapter_path)
    _raw_sha256(snapshot_hash, "snapshot_content_sha256")
    base_hash, model_config_hash = fingerprint_base_model_details(model_path)
    vocab, backend_tokenizer_json, vocab_size = _load_tokenizer_contract(model_path)
    tokenizer_config_hash = tokenizer_config_fingerprint(backend_tokenizer_json)
    vocab_hash, pair_count = tokenizer_vocab_fingerprint(vocab)
    if pair_count != vocab_size:
        raise TeacherLaunchError(
            "tokenizer get_vocab() pair count does not match backend get_vocab_size(true): "
            f"{pair_count} != {vocab_size}"
        )
    model_config = _strict_json_object(
        _read_strict_regular_file(model_path / "config.json", "config.json"),
        "config.json",
    )
    model_vocab_size = _model_vocab_size(model_config)
    if vocab_size > model_vocab_size:
        raise TeacherLaunchError(
            "backend tokenizer vocabulary entry count exceeds the model vocabulary: "
            f"{vocab_size} > {model_vocab_size}"
        )
    max_token_id = max(vocab.values())
    if max_token_id >= model_vocab_size:
        raise TeacherLaunchError(
            "backend tokenizer maximum token ID is outside the model vocabulary: "
            f"{max_token_id} >= {model_vocab_size}"
        )
    if args.adapter_path is not None:
        adapter = fingerprint_adapter(args.adapter_path, args.served_model_id)
        max_adapter_rank = adapter_max_rank(args.adapter_path)
    else:
        adapter = None
        max_adapter_rank = None
    vllm_version = _installed_vllm_version()
    runtime_versions = installed_runtime_versions(vllm_version)
    accelerator = probe_accelerator()
    runtime_content = capture_runtime_content()
    return {
        "base_model_sha256": base_hash,
        "snapshot_content_sha256": snapshot_hash,
        "model_config_sha256": model_config_hash,
        "tokenizer_vocab_sha256": vocab_hash,
        "tokenizer_config_sha256": tokenizer_config_hash,
        "adapter": adapter,
        "adapter_max_rank": max_adapter_rank,
        "vocab_size": model_vocab_size,
        "implementation": f"vllm:{vllm_version}",
        "runtime_versions": runtime_versions,
        "runtime_content_sha256": runtime_content.sha256,
        "accelerator": accelerator,
    }


def _execute(args: argparse.Namespace) -> int:
    _validate_requested_limits(args)
    if not args.manifest_only:
        validate_launch_environment(os.environ)
    runtime_environment = launch_environment()
    extra_args = validate_extra_vllm_args(args.vllm_args)
    real_launch = not args.manifest_only and not args.dry_run
    snapshot: ImmutableSnapshot | None = None
    effective_args = args
    try:
        if real_launch:
            if args.identity_input is not None:
                raise TeacherLaunchError("--identity-input is forbidden for a real launch")
            if args.model_path is None:
                raise TeacherLaunchError("--model-path is required for a real launch")
            snapshot = create_immutable_snapshot(
                args.model_path,
                args.adapter_path,
                args.snapshot_root,
            )
            effective_args = argparse.Namespace(**vars(args))
            effective_args.model_path = snapshot.model_path
            effective_args.adapter_path = snapshot.adapter_path
            effective_args.snapshot_content_sha256 = snapshot.manifest_sha256

        inputs = _identity_inputs(effective_args)
        adapter = inputs["adapter"]
        max_adapter_rank = inputs["adapter_max_rank"]
        row_width = min(args.max_top_k + 1, inputs["vocab_size"])
        theoretical_candidates = args.max_model_len * row_width
        max_prompt_logprob_candidates = args.max_prompt_logprob_candidates
        if max_prompt_logprob_candidates is None:
            max_prompt_logprob_candidates = min(
                DEFAULT_PROMPT_LOGPROB_CANDIDATES,
                theoretical_candidates,
            )
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
            snapshot_content_sha256=inputs["snapshot_content_sha256"],
            model_config_sha256=inputs["model_config_sha256"],
            max_top_k=args.max_top_k,
            max_model_len=args.max_model_len,
            max_prompt_logprob_candidates=max_prompt_logprob_candidates,
            adapter_enabled=adapter is not None,
            adapter_max_rank=max_adapter_rank,
            extra_args=extra_args,
            environment=runtime_environment,
            runtime_versions=inputs["runtime_versions"],
            runtime_content_sha256=inputs["runtime_content_sha256"],
            accelerator=inputs["accelerator"],
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
            max_prompt_logprob_candidates=max_prompt_logprob_candidates,
            implementation=inputs["implementation"],
            inference_config_sha256=inference_hash,
        )
        fingerprint = encode_system_fingerprint(identity)
        output: dict[str, Any] = {
            "identity": identity,
            "canonical_json": canonical_identity_json(identity).decode("utf-8"),
            "system_fingerprint": fingerprint,
            "runtime_content_sha256": inputs["runtime_content_sha256"],
        }
        command: list[str] | None = None
        if args.dry_run or not args.manifest_only:
            if args.model_path is None:
                raise TeacherLaunchError("--model-path is required to build a vLLM command")
            command = build_vllm_command(
                model_path=effective_args.model_path,
                served_model_id=args.served_model_id,
                adapter_path=effective_args.adapter_path,
                adapter_max_rank=max_adapter_rank,
                max_top_k=args.max_top_k,
                max_model_len=args.max_model_len,
                system_fingerprint=fingerprint,
                extra_args=extra_args,
            )
        if args.dry_run and command is not None:
            output["command"] = _redact_command(command)
            output["runtime_lora_updates"] = "disabled"
            output["path_mode"] = "source preview; real launches use a verified private snapshot"
        if args.manifest_only or args.dry_run:
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0

        assert command is not None
        assert snapshot is not None
        snapshot.verify()
        verify_child_runtime_contract(
            inputs["runtime_versions"],
            inputs["runtime_content_sha256"],
            runtime_environment,
        )
        print(
            json.dumps(
                {
                    "system_fingerprint": fingerprint,
                    "runtime_content_sha256": inputs["runtime_content_sha256"],
                    "snapshot": os.fspath(snapshot.path),
                    "snapshot_cleanup": "when the supervised vLLM process exits",
                }
            ),
            flush=True,
        )
        return run_vllm_child(
            command,
            runtime_environment,
            process_group_mode=args.process_group_mode,
        )
    finally:
        if snapshot is not None:
            snapshot.cleanup()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return _execute(args)
    except TeacherLaunchError as exc:
        print(f"vLLM teacher launch failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
