#!/usr/bin/env python3
"""Materialize a closed, symlink-free base model for CUDA serving comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Sequence

from model_fingerprint import ModelFingerprintError, fingerprint_model


SCHEMA = "kiln.cuda-serving-model-materialization.v1"
EXCLUDED_DIRECTORIES = {".cache", "adapters"}
READ_CHUNK_BYTES = 8 * 1024 * 1024


class MaterializationError(RuntimeError):
    """The source model cannot be represented by the closed serving input."""


def _absolute_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise MaterializationError(f"cannot inspect {label} {absolute}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise MaterializationError(f"{label} must be a non-symlink directory")
    return absolute


def source_files(path: Path) -> tuple[Path, list[str], list[str]]:
    root = _absolute_directory(path, "source model")
    files: list[str] = []
    excluded: list[str] = []
    try:
        entries = sorted(os.scandir(root), key=lambda entry: entry.name)
    except OSError as exc:
        raise MaterializationError(f"cannot enumerate source model: {exc}") from exc
    for entry in entries:
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise MaterializationError(
                f"cannot inspect source model entry {entry.name!r}: {exc}"
            ) from exc
        if stat.S_ISREG(metadata.st_mode):
            files.append(entry.name)
        elif stat.S_ISDIR(metadata.st_mode) and entry.name in EXCLUDED_DIRECTORIES:
            excluded.append(entry.name)
        elif stat.S_ISLNK(metadata.st_mode):
            raise MaterializationError(
                f"source model root contains symlink {entry.name!r}"
            )
        elif stat.S_ISDIR(metadata.st_mode):
            raise MaterializationError(
                f"source model root contains undeclared directory {entry.name!r}"
            )
        else:
            raise MaterializationError(
                f"source model root contains special entry {entry.name!r}"
            )
    if not files:
        raise MaterializationError("source model has no root regular files")
    return root, files, excluded


def _hash_file(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise MaterializationError(f"cannot open serving model file {path}: {exc}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise MaterializationError(f"serving model entry is not regular: {path}")
        while True:
            chunk = os.read(descriptor, READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        initial.st_dev,
        initial.st_ino,
        initial.st_size,
        initial.st_mtime_ns,
        initial.st_ctime_ns,
    )
    final_identity = (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
        final.st_ctime_ns,
    )
    if final_identity != identity or size != initial.st_size:
        raise MaterializationError(f"serving model file changed while hashed: {path}")
    return size, "sha256:" + digest.hexdigest()


def content_inventory(root: Path, names: list[str]) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for name in names:
        size, digest = _hash_file(root / name)
        rows.append({"path": name, "bytes": size, "sha256": digest})
    payload = json.dumps(
        rows,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return rows, "sha256:" + hashlib.sha256(payload).hexdigest()


def _validate_target_layout(
    target: Path,
    source: Path,
    expected_names: list[str],
) -> None:
    root = _absolute_directory(target, "serving model")
    metadata = root.stat(follow_symlinks=False)
    if metadata.st_mode & 0o222:
        raise MaterializationError("serving model directory must be read-only")
    try:
        entries = sorted(os.scandir(root), key=lambda entry: entry.name)
    except OSError as exc:
        raise MaterializationError(f"cannot enumerate serving model: {exc}") from exc
    if [entry.name for entry in entries] != expected_names:
        raise MaterializationError(
            "serving model entries disagree with source root regular files"
        )
    for entry in entries:
        metadata = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise MaterializationError(
                f"serving model entry must be regular: {entry.name!r}"
            )
        source_metadata = (source / entry.name).stat(follow_symlinks=False)
        if (
            metadata.st_dev,
            metadata.st_ino,
        ) != (
            source_metadata.st_dev,
            source_metadata.st_ino,
        ):
            raise MaterializationError(
                f"serving model entry is not the bound source hardlink: {entry.name!r}"
            )


def _link_file(source: Path, target: Path) -> None:
    try:
        source_metadata = source.stat(follow_symlinks=False)
        if not stat.S_ISREG(source_metadata.st_mode):
            raise MaterializationError(f"source model entry is not regular: {source}")
        os.link(source, target, follow_symlinks=False)
    except OSError as exc:
        raise MaterializationError(
            f"same-filesystem hardlink is required for serving model file "
            f"{source.name!r}: {exc}"
        ) from exc


def _remove_partial_target(target: Path) -> None:
    try:
        os.chmod(target, 0o700)
        for entry in os.scandir(target):
            os.unlink(entry.path)
        os.rmdir(target)
    except OSError:
        pass


def materialize(source: Path, target: Path, model_id: str) -> dict[str, Any]:
    source_root, names, excluded = source_files(source)
    target_root = Path(os.path.abspath(os.fspath(target.expanduser())))
    if target_root == source_root or target_root.is_relative_to(source_root):
        raise MaterializationError("serving model target must not be inside its source")
    try:
        source_model = fingerprint_model(source_root, model_id)
    except ModelFingerprintError as exc:
        raise MaterializationError(f"source model fingerprint failed: {exc}") from exc

    created = False
    if target_root.exists() or target_root.is_symlink():
        _validate_target_layout(target_root, source_root, names)
    else:
        parent = _absolute_directory(target_root.parent, "serving model parent")
        if parent.stat().st_mode & 0o022:
            raise MaterializationError("serving model parent is group/world writable")
        try:
            target_root.mkdir(mode=0o700)
            created = True
            for name in names:
                _link_file(source_root / name, target_root / name)
            os.chmod(target_root, 0o555)
            directory_fd = os.open(target_root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            if created:
                _remove_partial_target(target_root)
            raise
        _validate_target_layout(target_root, source_root, names)

    source_inventory, source_content_sha256 = content_inventory(source_root, names)
    target_inventory, target_content_sha256 = content_inventory(target_root, names)
    if target_inventory != source_inventory:
        raise MaterializationError("serving model content disagrees with its source")
    try:
        target_model = fingerprint_model(target_root, model_id)
    except ModelFingerprintError as exc:
        raise MaterializationError(f"serving model fingerprint failed: {exc}") from exc
    comparable_source_model = {**source_model, "path": target_model["path"]}
    if target_model != comparable_source_model:
        raise MaterializationError("serving and source model fingerprints disagree")
    if target_content_sha256 != source_content_sha256:
        raise MaterializationError("serving model aggregate content hash disagrees")
    return {
        "schema": SCHEMA,
        "source": str(source_root),
        "target": str(target_root),
        "model_id": model_id,
        "excluded_directories": excluded,
        "file_count": len(names),
        "logical_bytes": sum(row["bytes"] for row in target_inventory),
        "content_sha256": target_content_sha256,
        "model_fingerprint": target_model,
        "created": created,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = materialize(args.source, args.target, args.model_id)
        print(
            json.dumps(
                result,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    except MaterializationError as exc:
        print(f"error: CUDA serving model materialization: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
