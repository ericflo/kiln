#!/usr/bin/env python3
"""Compute the source identity used by local hardware qualification receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HASH_FORMAT = "kiln-source-tree-v1"

INCLUDED_FILES = {
    ".dockerignore",
    "Cargo.lock",
    "Cargo.toml",
    "assets/logo.png",
    "about.hbs",
    "about.toml",
    "desktop/Cargo.lock",
    "desktop/Cargo.toml",
    "desktop/build.rs",
    "desktop/tauri.conf.json",
    "deny.toml",
    "kiln.example.toml",
}
INCLUDED_PREFIXES = (
    "crates/",
    "deploy/",
    "desktop/capabilities/",
    "desktop/icons/",
    "desktop/src/",
    "desktop/ui/",
    "qualification/schema/",
    "qualification/workloads/",
    "scripts/",
)
EXCLUDED_PREFIXES = (
    "qualification/receipts/",
    "scripts/c2_artifacts/",
)


class SourceTreeHashError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceEntry:
    mode: str
    path: str


def _git(root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode(errors="replace").strip()
        raise SourceTreeHashError(
            f"git {' '.join(args)} failed" + (f": {detail}" if detail else "")
        ) from exc
    return completed.stdout


def is_source_path(path: str) -> bool:
    normalized = path.replace(os.sep, "/")
    if normalized in INCLUDED_FILES:
        return True
    if any(normalized.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    return any(normalized.startswith(prefix) for prefix in INCLUDED_PREFIXES)


def tracked_source_entries(root: Path = ROOT) -> list[SourceEntry]:
    raw = _git(root, "ls-files", "-s", "-z")
    entries: list[SourceEntry] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode = metadata.split(b" ", 1)[0].decode("ascii")
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise SourceTreeHashError("git ls-files returned an invalid record") from exc
        if is_source_path(path):
            entries.append(SourceEntry(mode=mode, path=path))
    entries.sort(key=lambda entry: entry.path.encode("utf-8"))
    return entries


def source_tree_hash(root: Path = ROOT) -> tuple[str, list[SourceEntry]]:
    entries = tracked_source_entries(root)
    if not entries:
        raise SourceTreeHashError("no tracked source inputs found")

    digest = hashlib.sha256()
    digest.update(HASH_FORMAT.encode("ascii") + b"\0")
    for entry in entries:
        path = root / entry.path
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise SourceTreeHashError(f"cannot read tracked source input {entry.path}: {exc}") from exc
        content_digest = hashlib.sha256(content).hexdigest()
        digest.update(entry.mode.encode("ascii") + b"\0")
        digest.update(entry.path.encode("utf-8") + b"\0")
        digest.update(str(len(content)).encode("ascii") + b"\0")
        digest.update(content_digest.encode("ascii") + b"\0")
    return f"sha256:{digest.hexdigest()}", entries


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Git worktree root (default: repository containing this script)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the hash format, digest, file count, and input paths as JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        digest, entries = source_tree_hash(args.root.resolve())
    except SourceTreeHashError as exc:
        print(f"source-tree hash failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "format": HASH_FORMAT,
                    "sha256": digest,
                    "file_count": len(entries),
                    "files": [entry.path for entry in entries],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
