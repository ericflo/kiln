#!/usr/bin/env python3
"""Fingerprint the exact model inputs used by qualification receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable


INDEX_FILENAME = "model.safetensors.index.json"
SINGLE_WEIGHT_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"
TOKENIZER_FILENAME = "tokenizer.json"
CHAT_TEMPLATE_FILENAME = "chat_template.jinja"
TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"
READ_CHUNK_BYTES = 8 * 1024 * 1024
MIB = 1024 * 1024
MIN_READ_MIB_PER_SECOND = 1
MAX_READ_MIB_PER_SECOND = 16_384
MAX_RATE_LIMIT_SLEEP_SECONDS = 0.025


class ModelFingerprintError(RuntimeError):
    """Raised when a model directory cannot be fingerprinted safely."""


class _ReadRateLimiter:
    def __init__(
        self,
        max_mib_per_second: int | None,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if max_mib_per_second is not None and (
            isinstance(max_mib_per_second, bool)
            or not isinstance(max_mib_per_second, int)
            or not MIN_READ_MIB_PER_SECOND
            <= max_mib_per_second
            <= MAX_READ_MIB_PER_SECOND
        ):
            raise ModelFingerprintError(
                "max read rate must be an integer in "
                f"{MIN_READ_MIB_PER_SECOND}..={MAX_READ_MIB_PER_SECOND} MiB/s"
            )
        self.max_mib_per_second = max_mib_per_second
        self.total_bytes = 0
        self._bytes_per_second = (
            max_mib_per_second * MIB
            if max_mib_per_second is not None
            else None
        )
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._started: float | None = None

    def account(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ModelFingerprintError("fingerprint read byte count must not be negative")
        self.total_bytes += byte_count
        if self._bytes_per_second is None:
            return
        if self._started is None:
            self._started = self._clock()
        deadline = self._started + self.total_bytes / self._bytes_per_second
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                return
            self._sleeper(min(remaining, MAX_RATE_LIMIT_SLEEP_SECONDS))


def _wait_for_start_gate(path: Path | None, timeout_seconds: float = 60.0) -> None:
    if path is None:
        return
    if not path.is_absolute():
        raise ModelFingerprintError("--start-gate must be absolute")
    deadline = time.monotonic() + timeout_seconds
    while True:
        if path.is_symlink():
            raise ModelFingerprintError("start gate must not be a symlink")
        if path.is_file():
            try:
                payload = path.read_bytes()
            except OSError as exc:
                raise ModelFingerprintError(f"cannot read start gate: {exc}") from exc
            if payload != b"go\n":
                raise ModelFingerprintError("start gate payload must equal 'go\\n'")
            return
        if path.exists():
            raise ModelFingerprintError("start gate must be a regular file")
        if time.monotonic() >= deadline:
            raise ModelFingerprintError(
                f"start gate was not released within {timeout_seconds:.3f} seconds"
            )
        time.sleep(0.01)


@dataclass
class _OpenInput:
    path: Path
    relative_path: str
    fd: int
    initial_stat: os.stat_result
    read_rate_limiter: _ReadRateLimiter
    observed_hash: str | None = None

    def read_bytes(self) -> bytes:
        os.lseek(self.fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(self.fd, READ_CHUNK_BYTES)
            if not chunk:
                payload = b"".join(chunks)
                self.observed_hash = f"sha256:{hashlib.sha256(payload).hexdigest()}"
                return payload
            self.read_rate_limiter.account(len(chunk))
            chunks.append(chunk)

    def _current_hash(self) -> str:
        os.lseek(self.fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(self.fd, READ_CHUNK_BYTES)
            if not chunk:
                return f"sha256:{digest.hexdigest()}"
            self.read_rate_limiter.account(len(chunk))
            digest.update(chunk)

    def hash(self) -> str:
        digest = self._current_hash()
        self.observed_hash = digest
        return digest

    def close(self) -> None:
        os.close(self.fd)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _reject_constant(value: str) -> None:
    raise ModelFingerprintError(f"non-finite JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ModelFingerprintError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _decode_utf8(payload: bytes, filename: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelFingerprintError(f"{filename} is not valid UTF-8: {exc}") from exc


def _parse_json_object(payload: bytes, filename: str) -> dict[str, Any]:
    try:
        value = json.loads(
            _decode_utf8(payload, filename),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ModelFingerprintError:
        raise
    except json.JSONDecodeError as exc:
        raise ModelFingerprintError(f"failed to parse {filename}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelFingerprintError(f"{filename} must contain a JSON object")
    return value


def _safe_relative_path(raw: str, *, context: str) -> str:
    if not raw:
        raise ModelFingerprintError(f"{context} must not be empty")
    if "\x00" in raw:
        raise ModelFingerprintError(f"{context} contains a NUL byte")
    if "\\" in raw:
        raise ModelFingerprintError(f"{context} must use '/' as its path separator")
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ModelFingerprintError(f"{context} is not valid Unicode: {exc}") from exc
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ModelFingerprintError(f"{context} must stay inside the model directory: {raw!r}")
    if "." in path.parts or str(path) != raw:
        raise ModelFingerprintError(f"{context} must be a normalized relative path: {raw!r}")
    return raw


def _absolute_model_root(model_path: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(model_path)))
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise ModelFingerprintError(f"cannot inspect model path {root}: {exc}") from exc
    if stat.S_ISLNK(root_stat.st_mode):
        raise ModelFingerprintError(f"model path must not be a symlink: {root}")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ModelFingerprintError(f"model path is not a directory: {root}")
    return root


def _assert_path_components(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            current_stat = current.lstat()
        except OSError as exc:
            raise ModelFingerprintError(
                f"model input {relative_path!r} cannot be inspected: {exc}"
            ) from exc
        if stat.S_ISLNK(current_stat.st_mode):
            raise ModelFingerprintError(f"model input must not use a symlink: {relative_path!r}")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(current_stat.st_mode):
            raise ModelFingerprintError(
                f"model input parent is not a directory: {relative_path!r}"
            )
    return current


def _path_exists_without_following(root: Path, relative_path: str) -> bool:
    try:
        (root / PurePosixPath(relative_path)).lstat()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ModelFingerprintError(
            f"cannot inspect model input {relative_path!r}: {exc}"
        ) from exc


def _open_regular(
    root: Path,
    relative_path: str,
    read_rate_limiter: _ReadRateLimiter | None = None,
) -> _OpenInput:
    relative_path = _safe_relative_path(relative_path, context="model input path")
    path = _assert_path_components(root, relative_path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ModelFingerprintError(f"cannot open model input {relative_path!r}: {exc}") from exc
    try:
        initial_stat = os.fstat(fd)
        if not stat.S_ISREG(initial_stat.st_mode):
            raise ModelFingerprintError(
                f"model input is not a regular file: {relative_path!r}"
            )
        return _OpenInput(
            path,
            relative_path,
            fd,
            initial_stat,
            read_rate_limiter or _ReadRateLimiter(None),
        )
    except BaseException:
        os.close(fd)
        raise


def _discover_index_shards(index: dict[str, Any]) -> list[str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ModelFingerprintError(f"{INDEX_FILENAME} is missing a 'weight_map' object")

    # The Rust loader preserves the first occurrence while loading. Receipt
    # output is sorted below so the content identity is independent of tensor
    # key order in the index.
    seen: set[str] = set()
    shards: list[str] = []
    for tensor_name, raw_filename in weight_map.items():
        if not isinstance(raw_filename, str):
            raise ModelFingerprintError(
                f"{INDEX_FILENAME} weight_map value for {tensor_name!r} is not a string"
            )
        filename = _safe_relative_path(
            raw_filename,
            context=f"{INDEX_FILENAME} weight_map value for {tensor_name!r}",
        )
        if filename not in seen:
            seen.add(filename)
            shards.append(filename)
    if not shards:
        raise ModelFingerprintError(f"{INDEX_FILENAME} has an empty weight_map")
    return shards


def _fallback_shards(root: Path) -> list[str]:
    try:
        names = [
            entry.name
            for entry in os.scandir(root)
            if PurePosixPath(entry.name).suffix == ".safetensors"
        ]
    except OSError as exc:
        raise ModelFingerprintError(f"cannot list model directory {root}: {exc}") from exc
    names.sort(key=lambda name: os.fsencode(name))
    if not names:
        raise ModelFingerprintError(f"no .safetensors files found in {root}")
    return names


def _verify_unchanged(root: Path, root_initial: os.stat_result, inputs: list[_OpenInput]) -> None:
    try:
        root_final = root.stat(follow_symlinks=False)
    except OSError as exc:
        raise ModelFingerprintError(f"cannot recheck model directory {root}: {exc}") from exc
    if _stat_identity(root_final) != _stat_identity(root_initial):
        raise ModelFingerprintError("model directory changed while it was being fingerprinted")

    for item in inputs:
        try:
            _assert_path_components(root, item.relative_path)
            descriptor_stat = os.fstat(item.fd)
            path_stat = item.path.stat(follow_symlinks=False)
        except OSError as exc:
            raise ModelFingerprintError(
                f"model input {item.relative_path!r} changed while it was being fingerprinted: {exc}"
            ) from exc
        identity = _stat_identity(item.initial_stat)
        if _stat_identity(descriptor_stat) != identity or _stat_identity(path_stat) != identity:
            raise ModelFingerprintError(
                f"model input {item.relative_path!r} changed while it was being fingerprinted"
            )
        if item.observed_hash is None:
            raise ModelFingerprintError(
                f"model input {item.relative_path!r} was opened but not fingerprinted"
            )
        if item._current_hash() != item.observed_hash:
            raise ModelFingerprintError(
                f"model input {item.relative_path!r} changed while it was being fingerprinted"
            )
        # Catch a concurrent write during the verification read itself. This
        # complements the digest comparison on filesystems whose timestamp
        # granularity cannot expose a fast same-length rewrite.
        if (
            _stat_identity(os.fstat(item.fd)) != identity
            or _stat_identity(item.path.stat(follow_symlinks=False)) != identity
        ):
            raise ModelFingerprintError(
                f"model input {item.relative_path!r} changed while it was being fingerprinted"
            )


def fingerprint_model(
    model_path: Path,
    model_id: str | None = None,
    *,
    max_read_mib_per_second: int | None = None,
    read_rate_limiter: _ReadRateLimiter | None = None,
) -> dict[str, Any]:
    """Return the strict receipt ``model`` object for a local checkpoint."""

    if max_read_mib_per_second is not None and read_rate_limiter is not None:
        raise ModelFingerprintError(
            "provide either max_read_mib_per_second or read_rate_limiter, not both"
        )
    if read_rate_limiter is None:
        read_rate_limiter = _ReadRateLimiter(max_read_mib_per_second)
    root = _absolute_model_root(model_path)
    resolved_id = root.name if model_id is None else model_id
    if not isinstance(resolved_id, str) or not resolved_id:
        raise ModelFingerprintError("model ID must not be empty")

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root_fd = os.open(root, directory_flags)
    except OSError as exc:
        raise ModelFingerprintError(f"cannot open model directory {root}: {exc}") from exc

    opened: list[_OpenInput] = []
    try:
        root_initial = os.fstat(root_fd)
        index_input: _OpenInput | None = None
        if _path_exists_without_following(root, INDEX_FILENAME):
            index_input = _open_regular(root, INDEX_FILENAME, read_rate_limiter)
            opened.append(index_input)
            shard_paths = _discover_index_shards(
                _parse_json_object(index_input.read_bytes(), INDEX_FILENAME)
            )
        elif _path_exists_without_following(root, SINGLE_WEIGHT_FILENAME):
            shard_paths = [SINGLE_WEIGHT_FILENAME]
        else:
            shard_paths = _fallback_shards(root)

        weight_inputs: list[_OpenInput] = []
        weight_identities: dict[tuple[int, int], str] = {}
        for relative_path in shard_paths:
            try:
                weight = _open_regular(root, relative_path, read_rate_limiter)
            except ModelFingerprintError as exc:
                if index_input is not None:
                    raise ModelFingerprintError(
                        f"shard {relative_path!r} referenced by {INDEX_FILENAME} is invalid: {exc}"
                    ) from exc
                raise
            physical_identity = (weight.initial_stat.st_dev, weight.initial_stat.st_ino)
            previous = weight_identities.get(physical_identity)
            if previous is not None:
                weight.close()
                raise ModelFingerprintError(
                    f"weight paths {previous!r} and {relative_path!r} reference the same file"
                )
            if weight.initial_stat.st_size <= 0:
                weight.close()
                raise ModelFingerprintError(f"weight file is empty: {relative_path!r}")
            weight_identities[physical_identity] = relative_path
            opened.append(weight)
            weight_inputs.append(weight)

        config = _open_regular(root, CONFIG_FILENAME, read_rate_limiter)
        opened.append(config)
        tokenizer = _open_regular(root, TOKENIZER_FILENAME, read_rate_limiter)
        opened.append(tokenizer)

        template: _OpenInput | None = None
        template_hash: str | None = None
        if _path_exists_without_following(root, CHAT_TEMPLATE_FILENAME):
            template = _open_regular(
                root, CHAT_TEMPLATE_FILENAME, read_rate_limiter
            )
            opened.append(template)
            template_bytes = template.read_bytes()
            _decode_utf8(template_bytes, CHAT_TEMPLATE_FILENAME)
            template_hash = f"sha256:{hashlib.sha256(template_bytes).hexdigest()}"
        elif _path_exists_without_following(root, TOKENIZER_CONFIG_FILENAME):
            tokenizer_config = _open_regular(
                root, TOKENIZER_CONFIG_FILENAME, read_rate_limiter
            )
            opened.append(tokenizer_config)
            tokenizer_config_value = _parse_json_object(
                tokenizer_config.read_bytes(), TOKENIZER_CONFIG_FILENAME
            )
            fallback_template = tokenizer_config_value.get("chat_template")
            if fallback_template is not None:
                if not isinstance(fallback_template, str):
                    raise ModelFingerprintError(
                        f"{TOKENIZER_CONFIG_FILENAME} chat_template must be a string"
                    )
                try:
                    fallback_template_bytes = fallback_template.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ModelFingerprintError(
                        f"{TOKENIZER_CONFIG_FILENAME} chat_template is not valid Unicode: {exc}"
                    ) from exc
                template_hash = f"sha256:{hashlib.sha256(fallback_template_bytes).hexdigest()}"

        weight_files = [
            {
                "path": item.relative_path,
                "sha256": item.hash(),
                "bytes": item.initial_stat.st_size,
            }
            for item in weight_inputs
        ]
        weight_files.sort(key=lambda item: item["path"].encode("utf-8"))
        result = {
            "id": resolved_id,
            "path": str(root),
            "weight_files": weight_files,
            "config_hash": config.hash(),
            "tokenizer_hash": tokenizer.hash(),
            "chat_template_hash": template_hash,
        }
        _verify_unchanged(root, root_initial, opened)
        return result
    finally:
        for item in reversed(opened):
            item.close()
        os.close(root_fd)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path, help="local model directory")
    parser.add_argument("--model-id", help="receipt model ID (default: model directory name)")
    parser.add_argument("--json", action="store_true", help="emit the receipt model object as JSON")
    parser.add_argument(
        "--max-read-mib-per-second",
        type=int,
        help="optional cumulative read-rate limit across both integrity passes",
    )
    parser.add_argument("--start-gate", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.max_read_mib_per_second is not None and not (
        MIN_READ_MIB_PER_SECOND
        <= args.max_read_mib_per_second
        <= MAX_READ_MIB_PER_SECOND
    ):
        parser.error(
            "max-read-mib-per-second must be in "
            f"{MIN_READ_MIB_PER_SECOND}..={MAX_READ_MIB_PER_SECOND}"
        )
    return args


def _print_human(value: dict[str, Any]) -> None:
    print(f"model: {value['id']}")
    print(f"path: {value['path']}")
    for weight in value["weight_files"]:
        print(f"weight: {weight['path']} {weight['bytes']} {weight['sha256']}")
    print(f"config: {value['config_hash']}")
    print(f"tokenizer: {value['tokenizer_hash']}")
    print(f"chat_template: {value['chat_template_hash'] or 'none'}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        _wait_for_start_gate(args.start_gate)
        value = fingerprint_model(
            args.model_path,
            args.model_id,
            max_read_mib_per_second=args.max_read_mib_per_second,
        )
    except ModelFingerprintError as exc:
        print(f"model fingerprint failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        _print_human(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
