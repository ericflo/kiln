#!/usr/bin/env python3
"""Enforce the reviewed physical-line budget for production source files."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = Path("contracts/production-file-budget-v1.json")
SUPPORTED_EXTENSIONS = {".css", ".js", ".rs"}
EXCLUDED_COMPONENTS = {"tests"}


class BudgetPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExceptionEntry:
    path: str
    max_lines: int
    rationale: str


@dataclass(frozen=True)
class Policy:
    max_lines: int
    exceptions: tuple[ExceptionEntry, ...]


@dataclass(frozen=True)
class SourceFile:
    path: str
    lines: int


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BudgetPolicyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise BudgetPolicyError(f"non-finite JSON number: {value}")


def _closed_keys(value: dict[str, object], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual == expected:
        return
    details: list[str] = []
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        details.append(f"missing keys: {', '.join(missing)}")
    if unknown:
        details.append(f"unknown keys: {', '.join(unknown)}")
    raise BudgetPolicyError(f"{context} has " + "; ".join(details))


def _positive_int(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise BudgetPolicyError(f"{context} must be a positive integer")
    return value


def _repo_path(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise BudgetPolicyError(f"{context} must be a non-empty string")
    if value.startswith("/") or "\\" in value or "\0" in value:
        raise BudgetPolicyError(f"{context} must be a normalized repository-relative path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise BudgetPolicyError(f"{context} must be a normalized repository-relative path")
    return value


def load_policy(path: Path) -> Policy:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BudgetPolicyError(f"cannot load policy {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BudgetPolicyError("policy root must be an object")
    _closed_keys(raw, {"schema_version", "max_production_file_lines", "exceptions"}, "policy")
    if raw["schema_version"] != 1:
        raise BudgetPolicyError("schema_version must be 1")
    max_lines = _positive_int(raw["max_production_file_lines"], "max_production_file_lines")

    raw_exceptions = raw["exceptions"]
    if not isinstance(raw_exceptions, list):
        raise BudgetPolicyError("exceptions must be an array")
    exceptions: list[ExceptionEntry] = []
    for index, item in enumerate(raw_exceptions):
        context = f"exceptions[{index}]"
        if not isinstance(item, dict):
            raise BudgetPolicyError(f"{context} must be an object")
        _closed_keys(item, {"path", "max_lines", "rationale"}, context)
        entry_path = _repo_path(item["path"], f"{context}.path")
        entry_max = _positive_int(item["max_lines"], f"{context}.max_lines")
        rationale = item["rationale"]
        if not isinstance(rationale, str) or len(rationale.strip()) < 40:
            raise BudgetPolicyError(f"{context}.rationale must contain at least 40 characters")
        if entry_max <= max_lines:
            raise BudgetPolicyError(
                f"{context}.max_lines must exceed the default production-file budget"
            )
        exceptions.append(ExceptionEntry(entry_path, entry_max, rationale.strip()))

    paths = [entry.path for entry in exceptions]
    if paths != sorted(set(paths), key=lambda value: value.encode("utf-8")):
        raise BudgetPolicyError("exceptions must be sorted by unique path")
    return Policy(max_lines, tuple(exceptions))


def is_production_source(path: Path) -> bool:
    parts = path.parts
    return (
        len(parts) >= 4
        and parts[0] == "crates"
        and "src" in parts[2:]
        and path.suffix in SUPPORTED_EXTENSIONS
        and not any(part in EXCLUDED_COMPONENTS for part in parts)
    )


def physical_line_count(path: Path) -> int:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise BudgetPolicyError(f"cannot read production source {path}: {exc}") from exc
    if not content:
        return 0
    return content.count(b"\n") + int(not content.endswith(b"\n"))


def source_files(root: Path) -> list[SourceFile]:
    crates = root / "crates"
    if not crates.is_dir():
        raise BudgetPolicyError(f"production source root does not exist: {crates}")
    files: list[SourceFile] = []
    for candidate in crates.rglob("*"):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        relative = candidate.relative_to(root)
        if is_production_source(relative):
            files.append(SourceFile(relative.as_posix(), physical_line_count(candidate)))
    files.sort(key=lambda item: item.path.encode("utf-8"))
    return files


def violations(files: list[SourceFile], policy: Policy) -> list[str]:
    by_path = {item.path: item for item in files}
    exceptions = {item.path: item for item in policy.exceptions}
    errors: list[str] = []

    for exception in policy.exceptions:
        source = by_path.get(exception.path)
        if source is None:
            errors.append(f"exception path is missing or outside production scope: {exception.path}")
            continue
        if source.lines <= policy.max_lines:
            errors.append(
                f"stale exception {source.path}: {source.lines} lines is within the "
                f"{policy.max_lines}-line default; remove the exception"
            )
        elif source.lines < exception.max_lines:
            errors.append(
                f"exception ceiling has headroom for {source.path}: {source.lines} lines "
                f"is below the reviewed ceiling of {exception.max_lines}; lower max_lines"
            )

    for source in files:
        exception = exceptions.get(source.path)
        limit = exception.max_lines if exception is not None else policy.max_lines
        if source.lines > limit:
            kind = "reviewed exception" if exception is not None else "default budget"
            errors.append(
                f"{source.path}: {source.lines} lines exceeds {kind} of {limit}; "
                "split the file or update the reviewed policy with a specific rationale"
            )
    return errors


def check(root: Path, policy_path: Path) -> tuple[Policy, list[SourceFile]]:
    policy = load_policy(policy_path)
    files = source_files(root)
    errors = violations(files, policy)
    if errors:
        raise BudgetPolicyError("production file budget failed:\n" + "\n".join(f"- {item}" for item in errors))
    return policy, files


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--policy", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    policy_path = args.policy.resolve() if args.policy else root / DEFAULT_POLICY
    try:
        policy, files = check(root, policy_path)
    except BudgetPolicyError as exc:
        print(exc, file=sys.stderr)
        return 1
    oversized = sum(item.lines > policy.max_lines for item in files)
    print(
        f"production file budget passed: {len(files)} files, "
        f"{policy.max_lines}-line default, {oversized} reviewed exceptions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
