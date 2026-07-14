#!/usr/bin/env python3
"""Inventory and ratchet tests that inspect implementation source as text."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = Path("contracts/source-parsing-test-inventory-v1.json")
REPORT_PATH = Path("docs/VERIFICATION_TEST_INVENTORY.md")
RATCHET_KEYS = (
    "max_test_count",
    "max_read_site_count",
    "max_text_assertion_count",
)
CLASSIFICATION_GUIDANCE = {
    "generated_artifact_text": (
        "Parse the canonical JSON/schema and assert typed fields; reserve rendered-text checks "
        "for the documentation renderer."
    ),
    "implementation_source_text": (
        "Replace with compile-time trait/type constraints, injected runtime behavior, property "
        "tests, or structured ownership metadata consumed by production."
    ),
    "qualification_driver_source_text": (
        "Exercise a public helper or CLI preflight and assert its structured output instead of "
        "matching the driver's Python source."
    ),
}
SOURCE_SUFFIXES = (
    ".c",
    ".cc",
    ".comp",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".html",
    ".js",
    ".mjs",
    ".py",
    ".rs",
    ".sh",
)
RUST_RAW_STRING_RE = re.compile(r"(?:br|rb|r)(?P<hashes>#{0,16})\"")


class InventoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class FunctionSpan:
    name: str
    start: int
    body_start: int
    body_end: int


def _mask_rust(source: str) -> str:
    """Blank strings and comments while retaining byte offsets and newlines."""
    chars = list(source)
    length = len(source)
    index = 0

    def blank(start: int, end: int) -> None:
        for offset in range(start, end):
            if chars[offset] != "\n":
                chars[offset] = " "

    while index < length:
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            end = length if end < 0 else end
            blank(index, end)
            index = end
            continue
        if source.startswith("/*", index):
            depth = 1
            cursor = index + 2
            while cursor < length and depth:
                if source.startswith("/*", cursor):
                    depth += 1
                    cursor += 2
                elif source.startswith("*/", cursor):
                    depth -= 1
                    cursor += 2
                else:
                    cursor += 1
            blank(index, cursor)
            index = cursor
            continue

        raw = RUST_RAW_STRING_RE.match(source, index)
        if raw is not None:
            hashes = raw.group("hashes")
            delimiter = '"' + hashes
            content_start = raw.end()
            close = source.find(delimiter, content_start)
            end = length if close < 0 else close + len(delimiter)
            blank(index, end)
            index = end
            continue
        if source[index] == '"':
            cursor = index + 1
            while cursor < length:
                if source[cursor] == "\\":
                    cursor += 2
                    continue
                cursor += 1
                if source[cursor - 1] == '"':
                    break
            blank(index, min(cursor, length))
            index = cursor
            continue
        if source[index] == "'":
            cursor = index + 1
            escaped = False
            while cursor < min(length, index + 12):
                character = source[cursor]
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == "'":
                    cursor += 1
                    blank(index, cursor)
                    index = cursor
                    break
                elif character == "\n":
                    break
                cursor += 1
            else:
                index += 1
            if index == cursor:
                continue
        index += 1
    return "".join(chars)


def _matching_brace(masked: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise InventoryError("unterminated Rust function body")


def _rust_test_spans(source: str) -> list[FunctionSpan]:
    masked = _mask_rust(source)
    attributes = list(
        re.finditer(
            r"#\s*\[\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*::\s*)*test"
            r"(?:\s*\([^]]*\))?\s*\]",
            masked,
        )
    )
    spans = []
    for offset, attribute in enumerate(attributes):
        limit = attributes[offset + 1].start() if offset + 1 < len(attributes) else len(masked)
        function = re.search(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\b", masked[attribute.end():limit])
        if function is None:
            continue
        function_start = attribute.end() + function.start()
        opening = masked.find("{", attribute.end() + function.end(), limit)
        if opening < 0:
            continue
        closing = _matching_brace(masked, opening)
        spans.append(FunctionSpan(function.group(1), function_start, opening + 1, closing))
    return spans


def _rust_function_spans(source: str) -> list[FunctionSpan]:
    masked = _mask_rust(source)
    spans = []
    for function in re.finditer(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\b", masked):
        opening = masked.find("{", function.end())
        terminator = masked.find(";", function.end())
        if opening < 0 or (terminator >= 0 and terminator < opening):
            continue
        try:
            closing = _matching_brace(masked, opening)
        except InventoryError:
            continue
        spans.append(FunctionSpan(function.group(1), function.start(), opening + 1, closing))
    return spans


def _balanced_call(source: str, masked: str, opening: int) -> str:
    depth = 0
    for index in range(opening, len(masked)):
        if masked[index] == "(":
            depth += 1
        elif masked[index] == ")":
            depth -= 1
            if depth == 0:
                return source[opening + 1:index]
    return source[opening + 1:]


def _compact(value: str, limit: int = 240) -> str:
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."


def _quoted_strings(value: str) -> list[str]:
    return [match.group(1) for match in re.finditer(r'"([^"\n]+)"', value)]


def _source_expression(value: str) -> bool:
    lowered = value.casefold()
    if any(_implementation_target(target) for target in _quoted_strings(value)):
        return True
    return any(
        re.search(rf"\b{re.escape(marker)}\b", lowered)
        for marker in (
            "backend_dir",
            "crate_path",
            "forward_path",
            "helper_path",
            "opd_path",
            "source_path",
            "trainer_source",
        )
    ) or 'join("src' in lowered or 'join("crates' in lowered


def _implementation_target(target: str) -> bool:
    normalized = target.replace("\\", "/").casefold()
    if normalized.startswith(("contracts/", "qualification/schema/", "qualification/workloads/")) or any(
        segment in normalized
        for segment in (
            "/contracts/",
            "/fixtures/",
            "/test_fixtures/",
            "/qualification/schema/",
            "/qualification/workloads/",
        )
    ):
        return False
    return normalized.endswith(SOURCE_SUFFIXES)


def _resolved_include(root: Path, test_path: Path, expression: str) -> str:
    strings = _quoted_strings(expression)
    if len(strings) != 1:
        return _compact(expression)
    target = (test_path.parent / strings[0]).resolve()
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return _compact(expression)


def _classification(test_name: str, targets: list[str]) -> str:
    joined = " ".join(targets).casefold()
    if "generated_capability_report" in test_name or any(
        marker in joined for marker in ("backend-capability-report", "docs/", "contracts/")
    ):
        return "generated_artifact_text"
    if "qualification" in joined and ".py" in joined:
        return "qualification_driver_source_text"
    return "implementation_source_text"


def _text_assertion_count(body: str, language: str) -> int:
    if language == "rust":
        patterns = (
            r"\.contains\s*\(",
            r"\.find\s*\(",
            r"\.matches\s*\(",
            r"\.split(?:_once)?\s*\(",
            r"compact_body\s*\(",
        )
    else:
        patterns = (
            r"assert(?:In|NotIn|Regex|NotRegex)\s*\(",
            r"\bin\s+source\b",
            r"\bsource\.(?:find|count|split|startswith|endswith)\s*\(",
        )
    return sum(len(re.findall(pattern, body)) for pattern in patterns)


def _module_include_bindings(root: Path, source: str, path: Path) -> dict[str, str]:
    bindings = {}
    masked = _mask_rust(source)
    pattern = re.compile(
        r"\b(?:const|static)\s+([A-Za-z_][A-Za-z0-9_]*)[^=;]*=\s*include_str!\s*\((.*?)\)\s*;",
        re.S,
    )
    for match in pattern.finditer(masked):
        expression = source[match.start(2) : match.end(2)]
        bindings[match.group(1)] = _resolved_include(root, path, expression)
    return bindings


def _rust_read_sites(
    root: Path, body: str, path: Path, bindings: dict[str, str]
) -> list[dict[str, str]]:
    masked = _mask_rust(body)
    read_sites: list[dict[str, str]] = []
    for name, target in bindings.items():
        if _implementation_target(target) and re.search(
            rf"\b{re.escape(name)}\b", masked
        ):
            read_sites.append({"api": "module_include_str", "target": target})
    for match in re.finditer(r"\binclude_str!\s*\(", masked):
        opening = masked.find("(", match.start())
        expression = _balanced_call(body, masked, opening)
        target = _resolved_include(root, path, expression)
        if _implementation_target(target):
            read_sites.append({"api": "include_str", "target": target})
    read_pattern = re.compile(r"(?:(?:std\s*::\s*)?fs\s*::\s*)read_to_string\s*\(")
    for match in read_pattern.finditer(masked):
        opening = masked.find("(", match.start())
        expression = _balanced_call(body, masked, opening)
        if _source_expression(expression):
            read_sites.append({"api": "read_to_string", "target": _compact(expression)})
    unique = {(item["api"], item["target"]): item for item in read_sites}
    return [unique[key] for key in sorted(unique)]


def _called_helpers(body: str, helper_names: set[str]) -> set[str]:
    masked = _mask_rust(body)
    return {
        name
        for name in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", masked)
        if name in helper_names
    }


def scan_rust(root: Path) -> list[dict[str, object]]:
    entries = []
    for path in sorted((root / "crates").rglob("*.rs")):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise InventoryError(f"cannot read {path}: {exc}") from exc
        if "include_str!" not in source and "read_to_string" not in source:
            continue
        spans = _rust_test_spans(source)
        if not spans:
            continue
        bindings = _module_include_bindings(root, source, path)
        test_locations = {(span.name, span.start) for span in spans}
        helper_spans = [
            span
            for span in _rust_function_spans(source)
            if (span.name, span.start) not in test_locations
        ]
        helper_bodies = {
            span.name: source[span.body_start:span.body_end] for span in helper_spans
        }
        helper_names = set(helper_bodies)
        helper_sites = {
            name: {
                (item["api"], item["target"]): item
                for item in _rust_read_sites(root, body, path, bindings)
            }
            for name, body in helper_bodies.items()
        }
        helper_calls = {
            name: _called_helpers(body, helper_names) - {name}
            for name, body in helper_bodies.items()
        }
        callers: dict[str, set[str]] = {name: set() for name in helper_names}
        for caller, dependencies in helper_calls.items():
            for dependency in dependencies:
                callers[dependency].add(caller)
        pending = deque(helper_names)
        queued = set(helper_names)
        while pending:
            name = pending.popleft()
            queued.remove(name)
            inherited = dict(helper_sites[name])
            for dependency in helper_calls[name]:
                inherited.update(helper_sites[dependency])
            if inherited == helper_sites[name]:
                continue
            helper_sites[name] = inherited
            for caller in callers[name] - queued:
                pending.append(caller)
                queued.add(caller)
        relative = path.relative_to(root).as_posix()
        for span in spans:
            body = source[span.body_start:span.body_end]
            read_sites = _rust_read_sites(root, body, path, bindings)
            for helper in _called_helpers(body, helper_names):
                read_sites.extend(helper_sites[helper].values())
            unique = {
                (item["api"], item["target"]): item for item in read_sites
            }
            read_sites = [unique[key] for key in sorted(unique)]
            if not read_sites:
                continue
            targets = [item["target"] for item in read_sites]
            classification = _classification(span.name, targets)
            entries.append(
                {
                    "language": "rust",
                    "test_path": relative,
                    "test_name": span.name,
                    "classification": classification,
                    "read_sites": read_sites,
                    "text_assertion_count": _text_assertion_count(body, "rust"),
                    "replacement": CLASSIFICATION_GUIDANCE[classification],
                }
            )
    return entries


def _python_parent_expression(call: ast.Call) -> str:
    if not isinstance(call.func, ast.Attribute):
        return ""
    try:
        return ast.unparse(call.func.value)
    except (ValueError, RecursionError):
        return ""


def _python_source_read(expression: str) -> bool:
    lowered = expression.casefold()
    if "qualification_dir" in lowered and not any(
        marker in lowered for marker in ("schema", "workload", "fixture")
    ):
        return True
    return "root" in lowered and ("crates" in lowered or ".rs" in lowered)


def scan_python(root: Path) -> list[dict[str, object]]:
    entries = []
    test_root = root / "scripts/qualification/tests"
    for path in sorted(test_root.glob("test_*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise InventoryError(f"cannot parse {path}: {exc}") from exc
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            read_sites = []
            for child in ast.walk(node):
                if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
                    continue
                if child.func.attr not in {"read_text", "read_bytes"}:
                    continue
                expression = _python_parent_expression(child)
                if _python_source_read(expression):
                    read_sites.append(
                        {"api": child.func.attr, "target": _compact(expression)}
                    )
            unique = {
                (item["api"], item["target"]): item for item in read_sites
            }
            read_sites = [unique[key] for key in sorted(unique)]
            if not read_sites:
                continue
            segment = ast.get_source_segment(source, node) or ""
            targets = [item["target"] for item in read_sites]
            classification = _classification(node.name, targets)
            entries.append(
                {
                    "language": "python",
                    "test_path": path.relative_to(root).as_posix(),
                    "test_name": node.name,
                    "classification": classification,
                    "read_sites": read_sites,
                    "text_assertion_count": _text_assertion_count(segment, "python"),
                    "replacement": CLASSIFICATION_GUIDANCE[classification],
                }
            )
    return entries


def _summary(entries: list[dict[str, object]]) -> dict[str, object]:
    classifications = Counter(str(item["classification"]) for item in entries)
    languages = Counter(str(item["language"]) for item in entries)
    owners = Counter(str(item["test_path"]) for item in entries)
    return {
        "test_count": len(entries),
        "read_site_count": sum(len(item["read_sites"]) for item in entries),
        "text_assertion_count": sum(int(item["text_assertion_count"]) for item in entries),
        "by_classification": dict(sorted(classifications.items())),
        "by_language": dict(sorted(languages.items())),
        "by_owner": dict(sorted(owners.items())),
    }


def _initial_ratchet(summary: dict[str, object]) -> dict[str, int]:
    return {
        "max_test_count": int(summary["test_count"]),
        "max_read_site_count": int(summary["read_site_count"]),
        "max_text_assertion_count": int(summary["text_assertion_count"]),
    }


def build_inventory(root: Path, ratchet: dict[str, int] | None = None) -> dict[str, object]:
    entries = scan_rust(root) + scan_python(root)
    entries.sort(
        key=lambda item: (
            str(item["test_path"]).encode("utf-8"),
            str(item["test_name"]).encode("utf-8"),
        )
    )
    summary = _summary(entries)
    return {
        "schema_version": 1,
        "policy": {
            "scope": (
                "Rust #[test]/#[...::test] functions and Python qualification test methods "
                "that read production or qualification-driver implementation source as text."
            ),
            "interpretation": (
                "An inventory entry is a migration obligation, not evidence that the asserted "
                "behavior is correct. Counts may decrease but may not exceed the ratchet."
            ),
            "classifications": CLASSIFICATION_GUIDANCE,
        },
        "ratchet": ratchet or _initial_ratchet(summary),
        "summary": summary,
        "entries": entries,
    }


def load_contract(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise InventoryError(f"{path} must contain source-parsing inventory schema version 1")
    ratchet = value.get("ratchet")
    if not isinstance(ratchet, dict) or set(ratchet) != set(RATCHET_KEYS):
        raise InventoryError(f"{path}.ratchet must contain exactly {list(RATCHET_KEYS)}")
    for key in RATCHET_KEYS:
        if type(ratchet[key]) is not int or ratchet[key] < 0:
            raise InventoryError(f"{path}.ratchet.{key} must be a non-negative integer")
    return value


def enforce_ratchet(inventory: dict[str, object]) -> None:
    summary = inventory["summary"]
    ratchet = inventory["ratchet"]
    assert isinstance(summary, dict) and isinstance(ratchet, dict)
    mappings = {
        "max_test_count": "test_count",
        "max_read_site_count": "read_site_count",
        "max_text_assertion_count": "text_assertion_count",
    }
    failures = []
    for maximum, observed_key in mappings.items():
        observed = int(summary[observed_key])
        limit = int(ratchet[maximum])
        if observed > limit:
            failures.append(f"{observed_key} is {observed}, above ratchet {limit}")
    if failures:
        raise InventoryError("source-parsing test ratchet increased: " + "; ".join(failures))


def lowered_ratchet(inventory: dict[str, object]) -> dict[str, int]:
    summary = inventory["summary"]
    ratchet = inventory["ratchet"]
    assert isinstance(summary, dict) and isinstance(ratchet, dict)
    return {
        "max_test_count": min(int(ratchet["max_test_count"]), int(summary["test_count"])),
        "max_read_site_count": min(
            int(ratchet["max_read_site_count"]), int(summary["read_site_count"])
        ),
        "max_text_assertion_count": min(
            int(ratchet["max_text_assertion_count"]),
            int(summary["text_assertion_count"]),
        ),
    }


def render_report(inventory: dict[str, object]) -> str:
    summary = inventory["summary"]
    ratchet = inventory["ratchet"]
    entries = inventory["entries"]
    assert isinstance(summary, dict) and isinstance(ratchet, dict) and isinstance(entries, list)
    lines = [
        "# Verification Test Inventory",
        "",
        "> Generated by `python3 scripts/check_source_parsing_tests.py --write`; do not edit by hand.",
        "",
        "Kiln currently has tests that inspect implementation source as text. These tests can",
        "detect a spelling change while missing the behavior they claim to protect. This page",
        "is the exact migration queue and ratchet; an entry is technical debt, not correctness",
        "evidence.",
        "",
        "## Current baseline",
        "",
        f"- Tests: **{summary['test_count']}**",
        f"- Direct or module-bound source reads: **{summary['read_site_count']}**",
        f"- Text search/split assertions: **{summary['text_assertion_count']}**",
        f"- Rust tests: **{summary['by_language'].get('rust', 0)}**",
        f"- Python qualification tests: **{summary['by_language'].get('python', 0)}**",
        "",
        "The three limits are monotonic: `--write` lowers them after a migration but refuses",
        "to bless an increase. The contract is",
        "`contracts/source-parsing-test-inventory-v1.json`.",
        "",
        "## Replacement policy",
        "",
        "| Classification | Required replacement |",
        "| --- | --- |",
    ]
    for classification, guidance in CLASSIFICATION_GUIDANCE.items():
        lines.append(f"| `{classification}` | {guidance} |")
    lines.extend(
        [
            "",
            "A compile-only type constraint proves interface shape. An injected runtime test",
            "proves routing and failure behavior. A property/state-machine test proves transition",
            "invariants. Canonical JSON or schema validation proves generated metadata. Substring",
            "presence in a `.rs`, `.cu`, or driver file proves none of those things.",
            "",
            "## Ownership concentration",
            "",
            "| Test owner | Tests |",
            "| --- | ---: |",
        ]
    )
    owners = summary["by_owner"]
    assert isinstance(owners, dict)
    for owner, count in sorted(owners.items(), key=lambda item: (-int(item[1]), item[0])):
        lines.append(f"| `{owner}` | {count} |")
    lines.extend(
        [
            "",
            "## Migration queue",
            "",
            "| Test | Classification | Reads | Text assertions |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for entry in entries:
        label = f"{entry['test_path']}::{entry['test_name']}"
        lines.append(
            f"| `{label}` | `{entry['classification']}` | {len(entry['read_sites'])} | "
            f"{entry['text_assertion_count']} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "Run:",
            "",
            "```bash",
            "python3 scripts/check_source_parsing_tests.py",
            "```",
            "",
            "After replacing one or more entries, run the same command with `--write`. It",
            "regenerates the exact contract and report and lowers each applicable ceiling. If",
            "any count rises, regeneration fails; increasing a ceiling requires an explicit",
            "reviewed contract edit rather than a hidden command-line escape hatch.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_bytes(value: dict[str, object]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    contract_path = root / CONTRACT_PATH
    report_path = root / REPORT_PATH
    try:
        expected = load_contract(contract_path) if contract_path.is_file() else None
        ratchet = expected["ratchet"] if expected is not None else None
        assert ratchet is None or isinstance(ratchet, dict)
        actual = build_inventory(root, ratchet=ratchet)
        enforce_ratchet(actual)
        if args.write:
            actual["ratchet"] = lowered_ratchet(actual)
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            contract_path.write_bytes(_json_bytes(actual))
            report_path.write_text(render_report(actual), encoding="utf-8")
            print(
                f"wrote source-parsing inventory: {actual['summary']['test_count']} tests, "
                f"{actual['summary']['read_site_count']} reads, "
                f"{actual['summary']['text_assertion_count']} text assertions"
            )
            return 0
        if expected is None:
            raise InventoryError(f"missing contract: {contract_path}")
        if _json_bytes(actual) != contract_path.read_bytes():
            raise InventoryError("source-parsing contract is stale; run with --write after review")
        report = render_report(actual)
        if report != report_path.read_text(encoding="utf-8"):
            raise InventoryError("verification inventory report is stale; run with --write")
        print(
            f"source-parsing inventory matches ({actual['summary']['test_count']} tests, "
            f"{actual['summary']['read_site_count']} reads, "
            f"{actual['summary']['text_assertion_count']} text assertions)"
        )
        return 0
    except (InventoryError, OSError, UnicodeError, ValueError) as exc:
        print(f"source-parsing inventory failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
