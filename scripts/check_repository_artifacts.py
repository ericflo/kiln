#!/usr/bin/env python3
"""Enforce the checked-in artifact retention and file-size policy."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = Path("contracts/repository-artifact-policy-v1.json")
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
INDEX_MODES = {"100644", "100755", "120000", "160000"}


class ArtifactPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class LargeFileException:
    path: str
    bytes: int
    sha256: str
    rationale: str


@dataclass(frozen=True)
class Policy:
    forbidden_suffixes: tuple[str, ...]
    max_csv_bytes: int
    max_file_bytes: int
    exceptions: tuple[LargeFileException, ...]


@dataclass(frozen=True)
class IndexedPath:
    path: str
    mode: str
    oid: str
    object_type: str
    bytes: int


@dataclass(frozen=True)
class Violation:
    entry: IndexedPath
    reasons: tuple[str, ...]


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactPolicyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ArtifactPolicyError(f"non-finite JSON number: {value}")


def _closed_keys(value: dict[str, object], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"missing keys: {', '.join(missing)}")
    if unknown:
        details.append(f"unknown keys: {', '.join(unknown)}")
    raise ArtifactPolicyError(f"{context} has " + "; ".join(details))


def _positive_int(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise ArtifactPolicyError(f"{context} must be a positive integer")
    return value


def _repo_path(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactPolicyError(f"{context} must be a non-empty string")
    if value.startswith("/") or "\\" in value or "\0" in value:
        raise ArtifactPolicyError(f"{context} must be a normalized repository-relative path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ArtifactPolicyError(f"{context} must be a normalized repository-relative path")
    return value


def load_policy(path: Path) -> Policy:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactPolicyError(f"cannot load policy {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactPolicyError("policy root must be an object")
    _closed_keys(
        value,
        {
            "schema_version",
            "forbidden_artifact_suffixes",
            "max_csv_bytes",
            "max_tracked_file_bytes",
            "large_file_exceptions",
        },
        "policy",
    )
    if value["schema_version"] != 1:
        raise ArtifactPolicyError("schema_version must be 1")

    raw_suffixes = value["forbidden_artifact_suffixes"]
    if not isinstance(raw_suffixes, list) or not raw_suffixes:
        raise ArtifactPolicyError("forbidden_artifact_suffixes must be a non-empty array")
    suffixes: list[str] = []
    for index, suffix in enumerate(raw_suffixes):
        if (
            not isinstance(suffix, str)
            or not suffix.startswith(".")
            or suffix != suffix.lower()
            or "/" in suffix
            or "\\" in suffix
            or len(suffix) < 2
        ):
            raise ArtifactPolicyError(
                f"forbidden_artifact_suffixes[{index}] must be a lowercase suffix beginning with '.'"
            )
        suffixes.append(suffix)
    if suffixes != sorted(set(suffixes)):
        raise ArtifactPolicyError("forbidden_artifact_suffixes must be sorted and unique")

    max_csv_bytes = _positive_int(value["max_csv_bytes"], "max_csv_bytes")
    max_file_bytes = _positive_int(
        value["max_tracked_file_bytes"], "max_tracked_file_bytes"
    )
    if max_csv_bytes >= max_file_bytes:
        raise ArtifactPolicyError("max_csv_bytes must be less than max_tracked_file_bytes")

    raw_exceptions = value["large_file_exceptions"]
    if not isinstance(raw_exceptions, list):
        raise ArtifactPolicyError("large_file_exceptions must be an array")
    exceptions: list[LargeFileException] = []
    for index, item in enumerate(raw_exceptions):
        context = f"large_file_exceptions[{index}]"
        if not isinstance(item, dict):
            raise ArtifactPolicyError(f"{context} must be an object")
        _closed_keys(item, {"path", "bytes", "sha256", "rationale"}, context)
        exception_path = _repo_path(item["path"], f"{context}.path")
        exception_bytes = _positive_int(item["bytes"], f"{context}.bytes")
        sha256 = item["sha256"]
        rationale = item["rationale"]
        if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256):
            raise ArtifactPolicyError(f"{context}.sha256 must be sha256:<64 lowercase hex>")
        if not isinstance(rationale, str) or len(rationale.strip()) < 20:
            raise ArtifactPolicyError(f"{context}.rationale must contain at least 20 characters")
        exceptions.append(
            LargeFileException(exception_path, exception_bytes, sha256, rationale.strip())
        )
    paths = [item.path for item in exceptions]
    if paths != sorted(set(paths), key=lambda item: item.encode("utf-8")):
        raise ArtifactPolicyError("large_file_exceptions must be sorted by unique path")
    return Policy(tuple(suffixes), max_csv_bytes, max_file_bytes, tuple(exceptions))


def _run_git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode(errors="replace").strip()
        raise ArtifactPolicyError(
            f"git {' '.join(args)} failed" + (f": {detail}" if detail else "")
        ) from exc
    return completed.stdout


def indexed_paths(root: Path) -> list[IndexedPath]:
    staged: list[tuple[str, str, str]] = []
    for record in _run_git(root, "ls-files", "-s", "-z").split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            raw_mode, raw_oid, raw_stage = metadata.split(b" ")
            mode = raw_mode.decode("ascii")
            oid = raw_oid.decode("ascii")
            stage = raw_stage.decode("ascii")
            path = raw_path.decode("utf-8", errors="surrogateescape")
        except (ValueError, UnicodeError) as exc:
            raise ArtifactPolicyError("git ls-files returned an invalid index record") from exc
        if stage != "0":
            raise ArtifactPolicyError(f"tracked path {path!r} has unresolved index stage {stage}")
        if mode not in INDEX_MODES:
            raise ArtifactPolicyError(f"tracked path {path!r} has unsupported mode {mode}")
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid):
            raise ArtifactPolicyError(f"tracked path {path!r} has invalid object ID")
        staged.append((path, mode, oid))

    blob_oids = sorted({oid for _, mode, oid in staged if mode != "160000"})
    query = b"".join(oid.encode("ascii") + b"\n" for oid in blob_oids)
    output = _run_git(
        root,
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        input_bytes=query,
    )
    info: dict[str, tuple[str, int]] = {}
    for line in output.splitlines():
        parts = line.decode("ascii").split(" ")
        if len(parts) != 3 or not parts[2].isdigit():
            raise ArtifactPolicyError("git cat-file returned invalid object metadata")
        info[parts[0]] = (parts[1], int(parts[2]))
    if set(info) != set(blob_oids):
        raise ArtifactPolicyError("git cat-file did not describe every indexed blob")

    entries = []
    for path, mode, oid in staged:
        if mode == "160000":
            entries.append(IndexedPath(path, mode, oid, "commit", 0))
            continue
        object_type, size = info[oid]
        if object_type != "blob":
            raise ArtifactPolicyError(f"tracked path {path!r} does not resolve to a blob")
        entries.append(IndexedPath(path, mode, oid, object_type, size))
    entries.sort(key=lambda item: item.path.encode("utf-8", errors="surrogateescape"))
    return entries


def find_violations(
    entries: list[IndexedPath],
    policy: Policy,
    content_sha256: Callable[[IndexedPath], str] | None = None,
) -> list[Violation]:
    exceptions = {item.path: item for item in policy.exceptions}
    by_path = {entry.path: entry for entry in entries}
    reasons_by_path: dict[str, list[str]] = {}
    described_by_path: dict[str, IndexedPath] = {}
    for entry in entries:
        if entry.object_type != "blob":
            continue
        lowered = entry.path.casefold()
        reasons: list[str] = []
        matched_suffix = next(
            (suffix for suffix in policy.forbidden_suffixes if lowered.endswith(suffix)), None
        )
        if matched_suffix is not None:
            reasons.append(f"forbidden artifact suffix {matched_suffix}")
        if lowered.endswith(".csv") and entry.bytes > policy.max_csv_bytes:
            reasons.append(f"CSV exceeds {policy.max_csv_bytes} bytes")
        if entry.bytes > policy.max_file_bytes:
            exception = exceptions.get(entry.path)
            allowed = False
            if exception is not None and exception.bytes == entry.bytes and content_sha256 is not None:
                allowed = content_sha256(entry) == exception.sha256
            if not allowed:
                reasons.append(f"tracked file exceeds {policy.max_file_bytes} bytes")
        if reasons:
            reasons_by_path.setdefault(entry.path, []).extend(reasons)
            described_by_path[entry.path] = entry

    for exception in policy.exceptions:
        entry = by_path.get(exception.path)
        reason = None
        if entry is None:
            reason = "large-file exception path is not tracked"
        elif entry.object_type != "blob":
            reason = "large-file exception path is not a blob"
        elif entry.bytes <= policy.max_file_bytes:
            reason = "large-file exception is no longer necessary"
        elif entry.bytes != exception.bytes:
            reason = "large-file exception byte count does not match"
        elif content_sha256 is None or content_sha256(entry) != exception.sha256:
            reason = "large-file exception SHA-256 does not match"
        if reason is not None:
            described = entry or IndexedPath(exception.path, "100644", "0" * 40, "missing", 0)
            reasons_by_path.setdefault(exception.path, []).append(reason)
            described_by_path[exception.path] = described
    return [
        Violation(described_by_path[path], tuple(dict.fromkeys(reasons_by_path[path])))
        for path in sorted(
            reasons_by_path,
            key=lambda item: item.encode("utf-8", errors="surrogateescape"),
        )
    ]


class WorktreeHasher:
    def __init__(self, root: Path) -> None:
        self.root = root
        object_format = _run_git(root, "rev-parse", "--show-object-format").decode().strip()
        if object_format not in {"sha1", "sha256"}:
            raise ArtifactPolicyError(f"unsupported Git object format: {object_format}")
        self.object_format = object_format
        self.cache: dict[tuple[str, str], str] = {}

    def sha256(self, entry: IndexedPath) -> str:
        key = (entry.oid, entry.path)
        if key in self.cache:
            return self.cache[key]
        path = self.root / entry.path
        git_digest = hashlib.new(self.object_format)
        git_digest.update(f"blob {entry.bytes}\0".encode("ascii"))
        content_digest = hashlib.sha256()
        observed = 0
        try:
            info = path.lstat()
            chunks: Iterable[bytes]
            if entry.mode == "120000":
                if not stat.S_ISLNK(info.st_mode):
                    raise ArtifactPolicyError(f"indexed symlink {entry.path!r} is not a worktree symlink")
                chunks = (os.readlink(os.fsencode(path)),)
            else:
                if not stat.S_ISREG(info.st_mode):
                    raise ArtifactPolicyError(f"indexed file {entry.path!r} is not a regular worktree file")

                def file_chunks() -> Iterable[bytes]:
                    with path.open("rb") as handle:
                        while chunk := handle.read(1024 * 1024):
                            yield chunk

                chunks = file_chunks()
            for chunk in chunks:
                observed += len(chunk)
                git_digest.update(chunk)
                content_digest.update(chunk)
        except ArtifactPolicyError:
            raise
        except OSError as exc:
            raise ArtifactPolicyError(f"cannot hash tracked path {entry.path!r}: {exc}") from exc
        if observed != entry.bytes or git_digest.hexdigest() != entry.oid:
            raise ArtifactPolicyError(
                f"worktree content for {entry.path!r} does not match its indexed blob; stage or restore it first"
            )
        result = f"sha256:{content_digest.hexdigest()}"
        self.cache[key] = result
        return result


def write_archive(
    output: Path,
    root: Path,
    policy_path: Path,
    violations: list[Violation],
    hasher: WorktreeHasher,
) -> None:
    if output.exists():
        raise ArtifactPolicyError(f"archive output already exists: {output}")
    if not violations:
        raise ArtifactPolicyError("there are no policy offenders to archive")
    source_commit = _run_git(root, "rev-parse", "HEAD").decode("ascii").strip()
    policy_bytes = policy_path.read_bytes()
    reason_counts: collections.Counter[str] = collections.Counter()
    suffix_counts: collections.Counter[str] = collections.Counter()
    recorded = []
    for violation in violations:
        reason_counts.update(violation.reasons)
        suffix_counts[Path(violation.entry.path).suffix.casefold() or "<none>"] += 1
        recorded.append(
            {
                "path": violation.entry.path,
                "bytes": violation.entry.bytes,
                "sha256": hasher.sha256(violation.entry),
                "reasons": list(violation.reasons),
            }
        )
    value = {
        "schema_version": 1,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "policy": {
            "path": policy_path.relative_to(root).as_posix(),
            "sha256": f"sha256:{hashlib.sha256(policy_bytes).hexdigest()}",
        },
        "history_rewritten": False,
        "restoration_command": "git show '{source_commit}:{path}' > <ignored-output-path>",
        "summary": {
            "artifact_count": len(recorded),
            "total_bytes": sum(item["bytes"] for item in recorded),
            "reason_counts": dict(sorted(reason_counts.items())),
            "suffix_counts": dict(sorted(suffix_counts.items())),
        },
        "artifacts": recorded,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Git worktree root")
    parser.add_argument(
        "--policy", type=Path, default=DEFAULT_POLICY,
        help="policy path, relative to --root by default",
    )
    parser.add_argument(
        "--archive-current-offenders", type=Path, metavar="PATH",
        help="write a one-time path/size/SHA-256 manifest before removing offenders",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    policy_path = args.policy if args.policy.is_absolute() else root / args.policy
    policy_path = policy_path.resolve()
    try:
        try:
            policy_path.relative_to(root)
        except ValueError as exc:
            raise ArtifactPolicyError("--policy must resolve inside --root") from exc
        policy = load_policy(policy_path)
        entries = indexed_paths(root)
        hasher = WorktreeHasher(root)
        violations = find_violations(entries, policy, hasher.sha256)
        if args.archive_current_offenders is not None:
            output = args.archive_current_offenders
            if not output.is_absolute():
                output = root / output
            write_archive(output, root, policy_path, violations, hasher)
            try:
                output_label: Path | str = output.relative_to(root)
            except ValueError:
                output_label = output
            print(f"archived {len(violations)} policy offenders to {output_label}")
            return 0
        if violations:
            for violation in violations:
                print(
                    f"artifact policy violation: {violation.entry.path!r} "
                    f"({violation.entry.bytes} bytes): {'; '.join(violation.reasons)}",
                    file=sys.stderr,
                )
            print(
                f"repository artifact policy failed: {len(violations)} tracked paths violate "
                f"{policy_path.relative_to(root)}",
                file=sys.stderr,
            )
            return 1
        total_bytes = sum(item.bytes for item in entries if item.object_type == "blob")
        print(
            f"repository artifact policy passed: {len(entries)} tracked paths, {total_bytes} bytes; "
            f"CSV <= {policy.max_csv_bytes}, each file <= {policy.max_file_bytes}"
        )
        return 0
    except (ArtifactPolicyError, OSError, ValueError) as exc:
        print(f"repository artifact policy failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
