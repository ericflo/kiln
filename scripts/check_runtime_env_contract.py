#!/usr/bin/env python3
"""Ratchet direct environment access in crate-owned source files.

The scanner is intentionally dependency-free so the contract can run on the
inexpensive qualification runner. It tokenizes source before looking for calls;
comments and string contents therefore cannot manufacture false call sites.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "contracts" / "runtime-env-direct-reads-v1.json"
RUST_SUFFIXES = frozenset({".rs"})
NATIVE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh"})
SOURCE_SUFFIXES = RUST_SUFFIXES | NATIVE_SUFFIXES
RUST_PREFILTER_RE = re.compile(
    r"(?:\bstd\s*::\s*env\b|\benv\s*!|\boption_env\s*!|\benv_flag\b|"
    r"\benv_tristate\b|\bset_var\b|\bremove_var\b|\bvar_os\b|\bvars_os\b)"
)
NATIVE_PREFILTER_RE = re.compile(
    r"\b(?:getenv|secure_getenv|setenv|unsetenv|putenv|clearenv)\b"
)
RUST_RAW_RE = re.compile(r"(?:br|rb|r)(#{0,255})\"")
CPP_RAW_RE = re.compile(r"(?:u8|u|U|L)?R\"([^ ()\\\t\r\n]{0,16})\(")
CHARACTER_PREFIX_RE = re.compile(r"(?:b|u8|u|U|L)?'")
LIFETIME_RE = re.compile(r"'[A-Za-z_][A-Za-z0-9_]*")
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

RUST_ENV_APIS = {
    "var": "read",
    "var_os": "read",
    "vars": "read",
    "vars_os": "read",
    "set_var": "set",
    "remove_var": "remove",
}
RUST_HELPERS = frozenset({"env_flag", "env_tristate"})
NATIVE_ENV_APIS = {
    "getenv": "read",
    "secure_getenv": "read",
    "setenv": "set",
    "unsetenv": "remove",
    "putenv": "set",
    "clearenv": "clear",
}


class ScanError(ValueError):
    """Raised when a source file cannot be tokenized safely."""


@dataclass(frozen=True)
class Token:
    kind: str
    value: str


def _decode_escaped_string(text: str, start: int, source: str) -> tuple[str, int]:
    value: list[str] = []
    index = start + 1
    escapes = {"0": "\0", "n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"', "'": "'"}
    while index < len(text):
        char = text[index]
        if char == '"':
            return "".join(value), index + 1
        if char != "\\":
            value.append(char)
            index += 1
            continue

        index += 1
        if index >= len(text):
            break
        escaped = text[index]
        if escaped in escapes:
            value.append(escapes[escaped])
            index += 1
        elif escaped == "x" and index + 2 < len(text):
            digits = text[index + 1 : index + 3]
            try:
                value.append(chr(int(digits, 16)))
            except ValueError as error:
                raise ScanError(f"{source}: invalid hexadecimal string escape") from error
            index += 3
        elif escaped == "u" and index + 1 < len(text) and text[index + 1] == "{":
            close = text.find("}", index + 2)
            if close < 0:
                raise ScanError(f"{source}: unterminated Unicode string escape")
            digits = text[index + 2 : close].replace("_", "")
            try:
                value.append(chr(int(digits, 16)))
            except (ValueError, OverflowError) as error:
                raise ScanError(f"{source}: invalid Unicode string escape") from error
            index = close + 1
        elif escaped in "\r\n":
            if escaped == "\r" and index + 1 < len(text) and text[index + 1] == "\n":
                index += 1
            index += 1
            while index < len(text) and text[index] in " \t":
                index += 1
        else:
            # The exact decoded value only matters for literal environment keys.
            # Retaining an unfamiliar escape is safer than silently dropping it.
            value.extend(("\\", escaped))
            index += 1
    raise ScanError(f"{source}: unterminated string literal")


def _rust_raw_string(text: str, index: int) -> tuple[str, int] | None:
    match = RUST_RAW_RE.match(text, index)
    if not match:
        return None
    hashes = match.group(1)
    content_start = match.end()
    terminator = '"' + hashes
    close = text.find(terminator, content_start)
    if close < 0:
        return None
    return text[content_start:close], close + len(terminator)


def _cpp_raw_string(text: str, index: int) -> tuple[str, int] | None:
    match = CPP_RAW_RE.match(text, index)
    if not match:
        return None
    delimiter = match.group(1)
    content_start = match.end()
    terminator = ")" + delimiter + '"'
    close = text.find(terminator, content_start)
    if close < 0:
        return None
    return text[content_start:close], close + len(terminator)


def _quoted_prefix_length(text: str, index: int, language: str) -> int | None:
    prefixes = ("", "b", "c") if language == "rust" else ("", "u8", "u", "U", "L")
    for prefix in sorted(prefixes, key=len, reverse=True):
        if text.startswith(prefix + '"', index):
            return len(prefix)
    return None


def _skip_character_literal(text: str, index: int, language: str, source: str) -> int | None:
    prefix_match = CHARACTER_PREFIX_RE.match(text, index)
    if not prefix_match:
        return None
    quote = prefix_match.end() - 1
    if language == "rust":
        lifetime = LIFETIME_RE.match(text, quote)
        if lifetime and lifetime.end() >= len(text):
            return None
        if lifetime and text[lifetime.end()] != "'":
            return None
    cursor = quote + 1
    escaped = False
    while cursor < len(text):
        char = text[cursor]
        if char == "'" and not escaped:
            return cursor + 1
        if char in "\r\n" and not escaped:
            raise ScanError(f"{source}: unterminated character literal")
        if char == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
        cursor += 1
    raise ScanError(f"{source}: unterminated character literal")


def tokenize(text: str, language: str, source: str = "<source>") -> list[Token]:
    """Return lexical tokens while discarding comments and literal interiors."""
    tokens: list[Token] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        if text.startswith("//", index):
            newline = text.find("\n", index + 2)
            index = len(text) if newline < 0 else newline + 1
            continue
        if text.startswith("/*", index):
            cursor = index + 2
            depth = 1
            while cursor < len(text) and depth:
                if language == "rust" and text.startswith("/*", cursor):
                    depth += 1
                    cursor += 2
                elif text.startswith("*/", cursor):
                    depth -= 1
                    cursor += 2
                else:
                    cursor += 1
            if depth:
                raise ScanError(f"{source}: unterminated block comment")
            index = cursor
            continue

        raw = _rust_raw_string(text, index) if language == "rust" else _cpp_raw_string(text, index)
        if raw is not None:
            value, index = raw
            tokens.append(Token("string", value))
            continue

        prefix_length = _quoted_prefix_length(text, index, language)
        if prefix_length is not None:
            value, index = _decode_escaped_string(text, index + prefix_length, source)
            tokens.append(Token("string", value))
            continue

        character_end = _skip_character_literal(text, index, language, source)
        if character_end is not None:
            tokens.append(Token("character", "<character>"))
            index = character_end
            continue

        identifier = IDENTIFIER_RE.match(text, index)
        if identifier:
            value = identifier.group(0)
            tokens.append(Token("identifier", value))
            index = identifier.end()
            continue
        if text.startswith("::", index):
            tokens.append(Token("punctuation", "::"))
            index += 2
            continue
        tokens.append(Token("punctuation", text[index]))
        index += 1
    return tokens


def _use_statements(tokens: list[Token]) -> tuple[set[str], dict[str, str], set[int]]:
    namespace_aliases: set[str] = set()
    function_aliases: dict[str, str] = {}
    use_indexes: set[int] = set()
    for start, token in enumerate(tokens):
        if token.value != "use":
            continue
        end = start + 1
        brace_depth = 0
        while end < len(tokens):
            if tokens[end].value == "{":
                brace_depth += 1
            elif tokens[end].value == "}":
                brace_depth -= 1
            elif tokens[end].value == ";" and brace_depth == 0:
                break
            end += 1
        if end >= len(tokens):
            continue
        use_indexes.update(range(start, end + 1))
        values = [item.value for item in tokens[start + 1 : end]]
        if values[:1] == ["::"]:
            values = values[1:]
        if values[:3] != ["std", "::", "env"]:
            continue
        remainder = values[3:]
        if not remainder:
            namespace_aliases.add("env")
        elif len(remainder) == 2 and remainder[0] == "as":
            namespace_aliases.add(remainder[1])
        elif remainder[:2] == ["::", "{"] and remainder[-1:] == ["}"]:
            group = remainder[2:-1]
            item: list[str] = []
            for value in [*group, ","]:
                if value != ",":
                    item.append(value)
                    continue
                if item[:1] == ["self"]:
                    namespace_aliases.add(item[2] if len(item) == 3 and item[1] == "as" else "env")
                elif item and item[0] in RUST_ENV_APIS:
                    alias = item[2] if len(item) == 3 and item[1] == "as" else item[0]
                    function_aliases[alias] = item[0]
                item = []
        elif len(remainder) >= 2 and remainder[0] == "::" and remainder[1] in RUST_ENV_APIS:
            canonical = remainder[1]
            alias = remainder[3] if len(remainder) == 4 and remainder[2] == "as" else canonical
            function_aliases[alias] = canonical
    return namespace_aliases, function_aliases, use_indexes


def _first_argument(tokens: list[Token], open_paren: int) -> tuple[str, str]:
    expression: list[Token] = []
    stack: list[str] = []
    matching = {")": "(", "]": "[", "}": "{"}
    for token in tokens[open_paren + 1 :]:
        value = token.value
        if not stack and value in {",", ")"}:
            break
        if value in {"(", "[", "{"}:
            stack.append(value)
        elif value in matching:
            if stack and stack[-1] == matching[value]:
                stack.pop()
        expression.append(token)
    if not expression:
        return "all", "<all>"
    if len(expression) == 1 and expression[0].kind == "string":
        return "literal", expression[0].value
    rendered = " ".join(
        json.dumps(token.value, ensure_ascii=True) if token.kind == "string" else token.value
        for token in expression
    )
    return "expression", rendered


def _source_surface(relative: Path) -> str:
    if relative.name == "build.rs":
        return "build-script"
    if "tests" in relative.parts:
        return "integration-test"
    if "examples" in relative.parts:
        return "example"
    if "benches" in relative.parts:
        return "benchmark"
    return "source"


def _entry(
    relative: Path,
    language: str,
    api: str,
    operation: str,
    argument_kind: str,
    argument: str,
) -> tuple[str, ...]:
    surface = _source_surface(relative)
    if api in {"env!", "option_env!"}:
        phase = "compile-time"
    elif surface == "build-script":
        phase = "build-time"
    else:
        phase = "runtime"
    return (
        relative.as_posix(),
        language,
        surface,
        phase,
        api,
        operation,
        argument_kind,
        argument,
    )


def scan_rust(text: str, relative: Path) -> list[tuple[str, ...]]:
    tokens = tokenize(text, "rust", relative.as_posix())
    namespaces, direct_aliases, use_indexes = _use_statements(tokens)
    matches: dict[tuple[str, int], tuple[str, int]] = {}

    for index, token in enumerate(tokens):
        if index in use_indexes:
            continue
        values = [item.value for item in tokens[index : index + 7]]
        offset = 1 if values[:1] == ["::"] else 0
        if values[offset : offset + 4] == ["std", "::", "env", "::"]:
            api_index = index + offset + 4
            if api_index + 1 < len(tokens):
                api = tokens[api_index].value
                if api in RUST_ENV_APIS and tokens[api_index + 1].value == "(":
                    matches[(api, api_index + 1)] = (api, api_index + 1)

        if token.value in namespaces and index + 3 < len(tokens):
            api = tokens[index + 2].value
            if tokens[index + 1].value == "::" and api in RUST_ENV_APIS and tokens[index + 3].value == "(":
                matches[(api, index + 3)] = (api, index + 3)

        if token.value in direct_aliases and index + 1 < len(tokens) and tokens[index + 1].value == "(":
            api = direct_aliases[token.value]
            matches[(api, index + 1)] = (api, index + 1)

        if token.value in RUST_HELPERS and index + 1 < len(tokens) and tokens[index + 1].value == "(":
            if index == 0 or tokens[index - 1].value != "fn":
                matches[(token.value, index + 1)] = (token.value, index + 1)

        if token.value in {"env", "option_env"} and index + 2 < len(tokens):
            if tokens[index + 1].value == "!" and tokens[index + 2].value == "(":
                api = token.value + "!"
                matches[(api, index + 2)] = (api, index + 2)

    entries: list[tuple[str, ...]] = []
    for api, open_paren in matches.values():
        argument_kind, argument = _first_argument(tokens, open_paren)
        operation = RUST_ENV_APIS.get(api, "read")
        entries.append(_entry(relative, "rust", api, operation, argument_kind, argument))
    return entries


def scan_native(text: str, relative: Path) -> list[tuple[str, ...]]:
    tokens = tokenize(text, "native", relative.as_posix())
    entries: list[tuple[str, ...]] = []
    for index, token in enumerate(tokens[:-1]):
        api = token.value
        if api not in NATIVE_ENV_APIS or tokens[index + 1].value != "(":
            continue
        argument_kind, argument = _first_argument(tokens, index + 1)
        entries.append(
            _entry(relative, "native", api, NATIVE_ENV_APIS[api], argument_kind, argument)
        )
    return entries


def scan_tree(repo_root: Path) -> Counter[tuple[str, ...]]:
    source_root = repo_root / "crates"
    if not source_root.is_dir():
        raise ScanError(f"missing source root: {source_root}")
    calls: Counter[tuple[str, ...]] = Counter()
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        relevant = RUST_PREFILTER_RE.search(text) if path.suffix in RUST_SUFFIXES else NATIVE_PREFILTER_RE.search(text)
        if relevant is None:
            continue
        relative = path.relative_to(repo_root)
        found = scan_rust(text, relative) if path.suffix in RUST_SUFFIXES else scan_native(text, relative)
        calls.update(found)
    return calls


ENTRY_FIELDS = (
    "path",
    "language",
    "surface",
    "phase",
    "api",
    "operation",
    "argument_kind",
    "argument",
)


def _entry_dict(key: tuple[str, ...], count: int) -> dict[str, object]:
    return {**dict(zip(ENTRY_FIELDS, key, strict=True)), "count": count}


def build_contract(repo_root: Path) -> dict[str, object]:
    calls = scan_tree(repo_root)
    reads: list[dict[str, object]] = []
    mutations: list[dict[str, object]] = []
    for key, count in sorted(calls.items()):
        destination = reads if key[5] == "read" else mutations
        destination.append(_entry_dict(key, count))
    return {
        "schema_version": 1,
        "source_root": "crates",
        "source_extensions": sorted(SOURCE_SUFFIXES),
        "summary": {
            "read_call_sites": sum(int(item["count"]) for item in reads),
            "read_entries": len(reads),
            "process_mutation_call_sites": sum(int(item["count"]) for item in mutations),
            "process_mutation_entries": len(mutations),
        },
        "reads": reads,
        "process_mutations": mutations,
    }


def _contract_counter(contract: dict[str, object]) -> Counter[tuple[str, ...]]:
    calls: Counter[tuple[str, ...]] = Counter()
    for section in ("reads", "process_mutations"):
        entries = contract.get(section)
        if not isinstance(entries, list):
            raise ScanError(f"contract field {section!r} must be a list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ScanError(f"contract field {section!r} contains a non-object entry")
            try:
                key = tuple(str(entry[field]) for field in ENTRY_FIELDS)
                count = int(entry["count"])
            except (KeyError, TypeError, ValueError) as error:
                raise ScanError(f"invalid entry in contract field {section!r}") from error
            calls[key] += count
    return calls


def _format_delta(key: tuple[str, ...], count: int) -> str:
    values = dict(zip(ENTRY_FIELDS, key, strict=True))
    return (
        f"  {count:+d} {values['path']} [{values['api']} "
        f"{values['argument_kind']}={values['argument']!r}]"
    )


def check_contract(expected: dict[str, object], actual: dict[str, object]) -> bool:
    if expected == actual:
        return True
    expected_calls = _contract_counter(expected)
    actual_calls = _contract_counter(actual)
    added = actual_calls - expected_calls
    removed = expected_calls - actual_calls
    print("runtime environment access contract drifted", file=sys.stderr)
    if added:
        print("new direct accesses (centralize them or deliberately refresh the baseline):", file=sys.stderr)
        for key, count in sorted(added.items())[:50]:
            print(_format_delta(key, count), file=sys.stderr)
    if removed:
        print("baseline accesses no longer present (refresh the baseline):", file=sys.stderr)
        for key, count in sorted(removed.items())[:50]:
            print(_format_delta(key, -count), file=sys.stderr)
    if not added and not removed:
        print("contract metadata or ordering is stale; regenerate it", file=sys.stderr)
    return False


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="check the committed baseline (default)")
    action.add_argument("--write", action="store_true", help="replace the baseline with the current scan")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--contract", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    contract_path = args.contract or (
        DEFAULT_CONTRACT if repo_root == ROOT else repo_root / "contracts" / DEFAULT_CONTRACT.name
    )
    try:
        actual = build_contract(repo_root)
        if args.write:
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            contract_path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote {contract_path}")
            return 0
        expected = json.loads(contract_path.read_text(encoding="utf-8"))
        if not isinstance(expected, dict):
            raise ScanError(f"contract root must be an object: {contract_path}")
        if not check_contract(expected, actual):
            return 1
    except (OSError, UnicodeError, json.JSONDecodeError, ScanError) as error:
        print(f"runtime environment contract error: {error}", file=sys.stderr)
        return 2
    summary = actual["summary"]
    assert isinstance(summary, dict)
    print(
        "runtime environment contract matches "
        f"({summary['read_call_sites']} reads, "
        f"{summary['process_mutation_call_sites']} process mutations)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
