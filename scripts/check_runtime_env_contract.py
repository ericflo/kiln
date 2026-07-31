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
DEFAULT_REPORT = ROOT / "docs" / "RUNTIME_ENVIRONMENT_INVENTORY.md"
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


TEST_SURFACES = frozenset({"unit-test", "integration-test", "benchmark", "example"})
PUBLIC_RUNTIME_BOUNDARIES = frozenset(
    {
        "crates/kiln-server/src/config.rs",
        "crates/kiln-server/src/logging.rs",
    }
)
STARTUP_SAFETY_BOUNDARIES = frozenset(
    {
        "crates/kiln-memory/src/startup_environment.rs",
    }
)
CREDENTIAL_PROVIDER_BOUNDARIES = frozenset(
    {
        "crates/kiln-train/src/credential_provider.rs",
    }
)
CLOSED_RUNTIME_READ_BOUNDARY_SHAPES = {
    "startup_safety": {
        "path": "crates/kiln-memory/src/startup_environment.rs",
        "language": "rust",
        "surface": "source",
        "phase": "runtime",
        "api": "var_os",
        "operation": "read",
        "argument_kind": "expression",
        "argument": "name",
        "count": 1,
    },
    "credential_provider": {
        "path": "crates/kiln-train/src/credential_provider.rs",
        "language": "rust",
        "surface": "source",
        "phase": "runtime",
        "api": "var",
        "operation": "read",
        "argument_kind": "expression",
        "argument": "name",
        "count": 1,
    },
}
BUILD_PROVENANCE_BOUNDARIES = frozenset(
    {
        "crates/kiln-server/src/execution_provenance.rs",
        "crates/kiln-train/src/replay.rs",
        "crates/kiln-train/src/train_receipt.rs",
    }
)


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


def _literal_bindings(tokens: list[Token]) -> dict[str, str]:
    """Return unambiguous file-local string constants used as env names."""
    candidates: dict[str, set[str]] = {}
    for index, token in enumerate(tokens[:-1]):
        if token.value not in {"const", "static"}:
            continue
        name = tokens[index + 1]
        if name.kind != "identifier":
            continue
        cursor = index + 2
        while cursor < len(tokens) and tokens[cursor].value not in {"=", ";"}:
            cursor += 1
        if cursor >= len(tokens) or tokens[cursor].value != "=":
            continue
        cursor += 1
        while cursor < len(tokens) and tokens[cursor].value in {"&"}:
            cursor += 1
        if cursor < len(tokens) and tokens[cursor].kind == "string":
            candidates.setdefault(name.value, set()).add(tokens[cursor].value)
    return {
        name: next(iter(values))
        for name, values in candidates.items()
        if len(values) == 1
    }


def _first_argument(
    tokens: list[Token],
    open_paren: int,
    literal_bindings: dict[str, str] | None = None,
) -> tuple[str, str]:
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
    if (
        len(expression) == 1
        and expression[0].kind == "identifier"
        and literal_bindings is not None
        and expression[0].value in literal_bindings
    ):
        return "literal", literal_bindings[expression[0].value]
    rendered = " ".join(
        json.dumps(token.value, ensure_ascii=True) if token.kind == "string" else token.value
        for token in expression
    )
    return "expression", rendered


def _matching_delimiter(
    tokens: list[Token], start: int, opening: str, closing: str
) -> int | None:
    depth = 0
    for index in range(start, len(tokens)):
        if tokens[index].value == opening:
            depth += 1
        elif tokens[index].value == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _attribute_requires_test(values: list[str]) -> bool:
    if values == ["test"] or (len(values) >= 3 and values[-2:] == ["::", "test"]):
        return True
    # The repository's gated unit-test modules use #[cfg(test)]. Be
    # conservative for richer cfg expressions rather than labeling code that
    # can compile without `test` as test-only.
    return values == ["cfg", "(", "test", ")"]


def _test_only_token_indexes(tokens: list[Token]) -> set[int]:
    indexes: set[int] = set()
    cursor = 0
    while cursor + 2 < len(tokens):
        if tokens[cursor].value != "#" or tokens[cursor + 1].value != "[":
            cursor += 1
            continue
        attribute_end = _matching_delimiter(tokens, cursor + 1, "[", "]")
        if attribute_end is None:
            cursor += 1
            continue
        values = [token.value for token in tokens[cursor + 2 : attribute_end]]
        if not _attribute_requires_test(values):
            cursor = attribute_end + 1
            continue

        item_start = attribute_end + 1
        while item_start + 1 < len(tokens) and tokens[item_start].value == "#":
            nested_end = _matching_delimiter(tokens, item_start + 1, "[", "]")
            if nested_end is None:
                break
            item_start = nested_end + 1

        item_end = item_start
        while item_end < len(tokens) and tokens[item_end].value not in {"{", ";"}:
            item_end += 1
        if item_end < len(tokens) and tokens[item_end].value == "{":
            matched = _matching_delimiter(tokens, item_end, "{", "}")
            if matched is not None:
                item_end = matched
            else:
                # A cfg(test) module conventionally owns the remainder of a
                # source file. Tokenization deliberately discards literal
                # interiors and can therefore make macro-heavy test modules
                # appear one delimiter short; classify the conservative
                # remainder as test-only instead of production runtime.
                item_end = len(tokens) - 1
        if item_end < len(tokens):
            indexes.update(range(cursor, item_end + 1))
            cursor = item_end + 1
        else:
            cursor = attribute_end + 1
    return indexes


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
    surface_override: str | None = None,
) -> tuple[str, ...]:
    surface = surface_override or _source_surface(relative)
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
    literal_bindings = _literal_bindings(tokens)
    test_only_indexes = _test_only_token_indexes(tokens)
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
        argument_kind, argument = _first_argument(tokens, open_paren, literal_bindings)
        operation = RUST_ENV_APIS.get(api, "read")
        surface_override = "unit-test" if open_paren in test_only_indexes else None
        entries.append(
            _entry(
                relative,
                "rust",
                api,
                operation,
                argument_kind,
                argument,
                surface_override,
            )
        )
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


def _classification(entry: dict[str, object]) -> tuple[str, str]:
    path = str(entry["path"])
    surface = str(entry["surface"])
    phase = str(entry["phase"])
    if surface in TEST_SURFACES:
        return "test_only", f"{surface} source surface"
    if phase in {"build-time", "compile-time"}:
        return "build_time", f"{phase} environment access"
    if path in STARTUP_SAFETY_BOUNDARIES:
        return "startup_safety", "immutable external-driver startup safety snapshot"
    if path in CREDENTIAL_PROVIDER_BOUNDARIES:
        return "credential_provider", "narrow configured credential-provider adapter"
    if path in BUILD_PROVENANCE_BOUNDARIES:
        return "build_time", "immutable build/source provenance boundary"
    if path in PUBLIC_RUNTIME_BOUNDARIES:
        return "public_stable", "central typed startup configuration boundary"
    return (
        "experimental_debug",
        "runtime access outside the typed startup configuration boundary",
    )


def _entry_dict(key: tuple[str, ...], count: int) -> dict[str, object]:
    entry: dict[str, object] = {
        **dict(zip(ENTRY_FIELDS, key, strict=True)),
        "count": count,
    }
    classification, basis = _classification(entry)
    entry["classification"] = classification
    entry["classification_basis"] = basis
    return entry


def build_contract(repo_root: Path) -> dict[str, object]:
    calls = scan_tree(repo_root)
    reads: list[dict[str, object]] = []
    mutations: list[dict[str, object]] = []
    for key, count in sorted(calls.items()):
        destination = reads if key[5] == "read" else mutations
        destination.append(_entry_dict(key, count))
    classifications = (
        "public_stable",
        "startup_safety",
        "credential_provider",
        "experimental_debug",
        "build_time",
        "test_only",
    )
    read_classifications = {
        classification: sum(
            int(item["count"])
            for item in reads
            if item["classification"] == classification
        )
        for classification in classifications
    }
    mutation_classifications = {
        classification: sum(
            int(item["count"])
            for item in mutations
            if item["classification"] == classification
        )
        for classification in classifications
    }
    literal_kiln_reads = [
        item
        for item in reads
        if item["argument_kind"] == "literal" and str(item["argument"]).startswith("KILN_")
    ]
    literal_kiln_names_by_classification = {
        classification: len(
            {
                str(item["argument"])
                for item in literal_kiln_reads
                if item["classification"] == classification
            }
        )
        for classification in classifications
    }
    return {
        "schema_version": 2,
        "source_root": "crates",
        "source_extensions": sorted(SOURCE_SUFFIXES),
        "summary": {
            "read_call_sites": sum(int(item["count"]) for item in reads),
            "read_entries": len(reads),
            "read_call_sites_by_classification": read_classifications,
            "literal_kiln_read_call_sites": sum(
                int(item["count"]) for item in literal_kiln_reads
            ),
            "literal_kiln_read_names": len(
                {str(item["argument"]) for item in literal_kiln_reads}
            ),
            "literal_kiln_read_names_by_classification": (
                literal_kiln_names_by_classification
            ),
            "process_mutation_call_sites": sum(int(item["count"]) for item in mutations),
            "process_mutation_entries": len(mutations),
            "process_mutation_call_sites_by_classification": mutation_classifications,
        },
        "reads": reads,
        "process_mutations": mutations,
    }


def _markdown_code(value: str) -> str:
    return f"`{value.replace('`', '&#96;')}`"


def render_report(contract: dict[str, object]) -> str:
    summary = contract["summary"]
    reads = contract["reads"]
    mutations = contract["process_mutations"]
    assert isinstance(summary, dict)
    assert isinstance(reads, list)
    assert isinstance(mutations, list)

    read_classes = summary["read_call_sites_by_classification"]
    mutation_classes = summary["process_mutation_call_sites_by_classification"]
    literal_names_by_class = summary["literal_kiln_read_names_by_classification"]
    assert isinstance(read_classes, dict)
    assert isinstance(mutation_classes, dict)
    assert isinstance(literal_names_by_class, dict)

    labels = {
        "public_stable": "Public stable",
        "startup_safety": "Startup safety",
        "credential_provider": "Credential provider",
        "experimental_debug": "Experimental/debug migration",
        "build_time": "Build time/provenance",
        "test_only": "Test only",
    }
    lines = [
        "# Runtime Environment Inventory",
        "",
        "> Generated by `python3 scripts/check_runtime_env_contract.py --write`; do not edit by hand.",
        "",
        "This page answers a source-ownership question: **which code under `crates/`",
        "reads or mutates the process environment, and why is that access allowed?**",
        "It is exhaustive for the scanner's direct-read APIs; it is not a list of settings",
        "that operators should copy into a shell.",
        "",
        "The machine-readable source of truth is",
        "[`contracts/runtime-env-direct-reads-v1.json`](../contracts/runtime-env-direct-reads-v1.json);",
        "`python3 scripts/check_runtime_env_contract.py` rejects any source, classification,",
        "or generated-report drift.",
        "",
        "## Start here",
        "",
        "| If you need to know… | Use… |",
        "|---|---|",
        "| Which environment overrides are supported | The [Configuration Reference](CONFIGURATION.md), which owns typed fields, mechanically derived canonical names, defaults, validation, source precedence, and restart behavior |",
        "| Why production code reads a process variable directly | The classification and owner catalogs on this page |",
        "| Which old names are rejected or migrated | The Configuration Reference's migration index; absence from this direct-read catalog is not migration guidance |",
        "| What a qualification case passes to a child process | The workload and effective run-configuration artifacts; inherited or launcher-passthrough values are not necessarily direct reads under `crates/` |",
        "| Which values contributed to execution identity | [Execution identity and provenance](EXECUTION_PROVENANCE.md); the provenance scan hashes the effective `KILN_*` map and redacts sensitive values |",
        "",
        "A name appearing below is not automatically public. Most entries are compiler or",
        "build inputs, test controls, provenance reads, or closed safety boundaries. The",
        "only supported public overrides are the names documented by the typed",
        "configuration registry.",
        "",
        "External driver visibility and remapping variables are also not Kiln settings.",
        "Kiln snapshots only their presence at first accelerator validation and fails",
        "device identity closed when a remap would make the probe ambiguous. It does not",
        "turn those names into request-time policy or a device allowlist.",
        "",
        "## Current baseline",
        "",
        f"The scanner records **{summary['read_call_sites']} direct read call sites** and",
        f"**{summary['process_mutation_call_sites']} process-mutation call sites**. It can",
        f"statically name **{summary['literal_kiln_read_names']} distinct literal `KILN_*`",
        f"read names** across **{summary['literal_kiln_read_call_sites']} call sites**.",
        "Dynamically named reads remain listed separately and are classified by their owner",
        "boundary.",
        "",
        "| Ownership class | Read call sites | Literal `KILN_*` names | Mutation call sites |",
        "|---|---:|---:|---:|",
    ]
    for classification in (
        "public_stable",
        "startup_safety",
        "credential_provider",
        "experimental_debug",
        "build_time",
        "test_only",
    ):
        lines.append(
            "| "
            + labels[classification]
            + f" | {int(read_classes.get(classification, 0))}"
            + f" | {int(literal_names_by_class.get(classification, 0))}"
            + f" | {int(mutation_classes.get(classification, 0))} |"
        )

    lines.extend(
        [
            "",
            "The counts are call sites, not configuration-field counts. The central typed",
            "loader deliberately uses a small number of dynamic reads to resolve all public",
            "registry entries once. Conversely, compile-time code and tests can contribute",
            "many call sites without adding a single public runtime setting.",
            "",
            "## Classification policy",
            "",
            "| Class | Meaning and required disposition |",
            "|---|---|",
            "| Public stable | Access occurs only in the central typed startup/configuration boundary. Public support still requires an entry in the Configuration Reference. |",
            "| Startup safety | One dedicated boundary snapshots closed external driver visibility/remapping names at first accelerator identity validation before model upload. These are not Kiln settings; their presence fails device/probe identity closed and cannot become request-time policy. |",
            "| Credential provider | One dedicated adapter resolves only the secret variable named by typed, exact-origin credential configuration. Secret values must never enter logs, serialization, API state, receipts, or caches. |",
            "| Experimental/debug migration | Runtime source reads the process environment outside that boundary. Move real policy into typed immutable configuration; put retained diagnostics behind one explicit experimental profile; delete dead controls. |",
            "| Build time/provenance | A build script, compile-time macro, or immutable build/source provenance boundary owns the read. It must never become request-time policy. |",
            "| Test only | The access is in a unit-test, integration-test, benchmark, or example surface. Prefer scoped typed fixtures; serialize the few tests that must mutate process-global state. |",
            "",
            "`#[cfg(test)]` modules and `#[test]` functions are recognized as test surfaces,",
            "including tests colocated in production source files. Simple file-local string",
            "constants used as environment names are resolved before classification. Unknown",
            "dynamic expressions are retained verbatim rather than guessed.",
            "",
            "## Production migration owners",
            "",
            "These files contain the runtime accesses still outside the typed startup",
            "boundary. This table is the prioritized deletion/migration queue.",
            "",
            "| Owner | Read call sites | Literal `KILN_*` names |",
            "|---|---:|---:|",
        ]
    )
    owner_calls: Counter[str] = Counter()
    owner_names: dict[str, set[str]] = {}
    for entry in reads:
        if entry["classification"] != "experimental_debug":
            continue
        path = str(entry["path"])
        owner_calls[path] += int(entry["count"])
        if entry["argument_kind"] == "literal" and str(entry["argument"]).startswith("KILN_"):
            owner_names.setdefault(path, set()).add(str(entry["argument"]))
    for path, count in sorted(owner_calls.items(), key=lambda item: (-item[1], item[0])):
        lines.append(
            f"| {_markdown_code(path)} | {count} | {len(owner_names.get(path, set()))} |"
        )
    if not owner_calls:
        lines.append("| None; the migration queue is empty | 0 | 0 |")

    lines.extend(
        [
            "",
            "## Literal `KILN_*` catalog",
            "",
            "A name can appear in more than one class, such as a build control that is also",
            "asserted by a test. Paths are deduplicated; counts retain duplicate call sites.",
            "",
            "<details>",
            "<summary>Show every literal KILN_* direct read</summary>",
            "",
            "| Name | Class | Read call sites | Owners |",
            "|---|---|---:|---|",
        ]
    )
    catalog: dict[str, dict[str, object]] = {}
    for entry in reads:
        argument = str(entry["argument"])
        if entry["argument_kind"] != "literal" or not argument.startswith("KILN_"):
            continue
        item = catalog.setdefault(
            argument,
            {"classifications": set(), "count": 0, "paths": set()},
        )
        classifications = item["classifications"]
        paths = item["paths"]
        assert isinstance(classifications, set)
        assert isinstance(paths, set)
        classifications.add(str(entry["classification"]))
        paths.add(str(entry["path"]))
        item["count"] = int(item["count"]) + int(entry["count"])
    for name, item in sorted(catalog.items()):
        classifications = item["classifications"]
        paths = item["paths"]
        assert isinstance(classifications, set)
        assert isinstance(paths, set)
        rendered_classes = ", ".join(labels[value] for value in sorted(classifications))
        rendered_paths = ", ".join(_markdown_code(path) for path in sorted(paths))
        lines.append(
            f"| {_markdown_code(name)} | {rendered_classes} | {item['count']} | {rendered_paths} |"
        )
    lines.extend(["", "</details>"])

    lines.extend(
        [
            "",
            "## Dynamic read catalog",
            "",
            "These direct reads do not expose one literal name at the call site. They include",
            "central registry loops, the narrow credential adapter, startup safety snapshots,",
            "helper functions, and whole-environment provenance scans. A dynamic read does",
            "not mean “accept any setting”: its owner constrains which names it may resolve.",
            "The exact expression remains ratcheted so a helper cannot conceal source growth.",
            "",
            "<details>",
            "<summary>Show every non-literal direct read</summary>",
            "",
            "| Owner | API | Argument | Class | Call sites |",
            "|---|---|---|---|---:|",
        ]
    )
    for entry in reads:
        argument = str(entry["argument"])
        if entry["argument_kind"] == "literal" and argument.startswith("KILN_"):
            continue
        lines.append(
            f"| {_markdown_code(str(entry['path']))} | {_markdown_code(str(entry['api']))} "
            f"| {_markdown_code(argument)} | {labels[str(entry['classification'])]} "
            f"| {entry['count']} |"
        )
    lines.extend(["", "</details>"])

    lines.extend(
        [
            "",
            "## Mutation boundary",
            "",
            "Process-environment mutation is forbidden in production execution. The current",
            "inventory contains only test/example mutation and build-script toolchain setup;",
            "the classification check will fail if a production mutation is introduced.",
            "Tests should still migrate toward scoped typed configuration because process-global",
            "mutation creates ordering and parallelism hazards even when it is test-only.",
            "",
            "## Migration rule",
            "",
            "The repository check rejects every experimental/debug migration read even if the",
            "JSON baseline is regenerated. A new production read must instead be removed or",
            "belong to a reviewed closed boundary above. For each production owner, move a",
            "coherent policy family at once: define typed",
            "fields, derive canonical `KILN_<SECTION>_<FIELD>` compatibility inputs",
            "mechanically, validate once at startup, inject immutable policy, expose effective",
            "value/source/restart semantics, remove lower rereads, and lower this ratchet in the",
            "same commit. Do not promote one-off kernel flags into permanent public API merely",
            "to preserve them.",
            "",
            "Retired or unknown public spellings do not become safe because a test, build",
            "script, or provenance scan mentions them. Public recognition requires the typed",
            "configuration registry; every lower-level runtime reread remains forbidden.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_policy(contract: dict[str, object]) -> None:
    reads = contract.get("reads")
    if not isinstance(reads, list):
        raise ScanError("contract field 'reads' must be a list")
    unclassified_runtime = [
        entry for entry in reads if entry.get("classification") == "experimental_debug"
    ]
    if unclassified_runtime:
        rendered = ", ".join(
            f"{entry['path']}:{entry['api']}({entry['argument']})"
            for entry in unclassified_runtime[:10]
        )
        raise ScanError(
            "direct production runtime environment reads are forbidden outside the "
            f"typed startup, startup-safety, credential-provider, and provenance boundaries: {rendered}"
        )

    for classification, expected in CLOSED_RUNTIME_READ_BOUNDARY_SHAPES.items():
        actual = [
            {field: entry.get(field) for field in (*ENTRY_FIELDS, "count")}
            for entry in reads
            if entry.get("classification") == classification
        ]
        if actual != [expected]:
            raise ScanError(
                f"{classification.replace('_', '-')} direct-read boundary must remain "
                f"exactly one closed call shape; expected {expected!r}, got {actual!r}"
            )

    mutations = contract.get("process_mutations")
    if not isinstance(mutations, list):
        raise ScanError("contract field 'process_mutations' must be a list")
    production = [
        entry
        for entry in mutations
        if entry.get("classification") not in {"test_only", "build_time"}
    ]
    if production:
        rendered = ", ".join(
            f"{entry['path']}:{entry['api']}({entry['argument']})"
            for entry in production[:10]
        )
        raise ScanError(f"production process-environment mutation is forbidden: {rendered}")


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
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    contract_path = args.contract or (
        DEFAULT_CONTRACT if repo_root == ROOT else repo_root / "contracts" / DEFAULT_CONTRACT.name
    )
    report_path = args.report or (
        DEFAULT_REPORT if repo_root == ROOT else repo_root / "docs" / DEFAULT_REPORT.name
    )
    try:
        actual = build_contract(repo_root)
        validate_policy(actual)
        rendered_report = render_report(actual)
        if args.write:
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            contract_path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(rendered_report, encoding="utf-8")
            print(f"wrote {contract_path} and {report_path}")
            return 0
        expected = json.loads(contract_path.read_text(encoding="utf-8"))
        if not isinstance(expected, dict):
            raise ScanError(f"contract root must be an object: {contract_path}")
        if not check_contract(expected, actual):
            return 1
        committed_report = report_path.read_text(encoding="utf-8")
        if committed_report != rendered_report:
            print(
                "runtime environment inventory report drifted; regenerate it with --write",
                file=sys.stderr,
            )
            return 1
    except (OSError, UnicodeError, json.JSONDecodeError, ScanError) as error:
        print(f"runtime environment contract error: {error}", file=sys.stderr)
        return 2
    summary = actual["summary"]
    assert isinstance(summary, dict)
    print(
        "runtime environment contract matches "
        f"({summary['read_call_sites']} reads, "
        f"{summary['process_mutation_call_sites']} process mutations; "
        f"{summary['read_call_sites_by_classification'].get('experimental_debug', 0)} "
        "runtime migration reads)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
