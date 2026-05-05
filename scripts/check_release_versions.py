#!/usr/bin/env python3
"""Guard user-facing Kiln examples against stale release and CLI drift."""
from __future__ import annotations

import html
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SERVER_VERSION_RE = re.compile(r'^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"\s*$', re.MULTILINE)


def package_version(path: Path) -> str:
    text = path.read_text()
    match = SERVER_VERSION_RE.search(text)
    if not match:
        raise SystemExit(f"could not find package version in {path}")
    return match.group(1)


SERVER_VERSION = package_version(ROOT / "Cargo.toml")
DESKTOP_VERSION = package_version(ROOT / "desktop" / "Cargo.toml")

CURRENT_SERVER_SURFACES = [
    ROOT / "README.md",
    ROOT / "QUICKSTART.md",
    ROOT / "docs/site/index.html",
    ROOT / "docs/site/quickstart.html",
    ROOT / "docs/site/demo/SCRIPT.md",
    ROOT / "docs/site/demo/index.html",
]

DISALLOWED_CURRENT_SERVER_PATTERNS = [
    (re.compile(r"kiln-v[0-9]+\.[0-9]+\.[0-9]+"), "server release tags should use /releases/latest or a latest-release lookup"),
    (re.compile(r"kiln-[0-9]+\.[0-9]+\.[0-9]+-(?:x86_64|aarch64)"), "server release asset names should derive from KILN_VERSION"),
    (re.compile(r"ghcr\.io/ericflo/kiln-server:[0-9]+\.[0-9]+\.[0-9]+"), "Docker examples should use :latest or a computed KILN_VERSION"),
    (re.compile(rf"Version:\s*{re.escape(SERVER_VERSION)}"), "sample startup banners should use <workspace version>"),
]

DESKTOP_SURFACES = [
    ROOT / "README.md",
    ROOT / "QUICKSTART.md",
    ROOT / "desktop/README.md",
    ROOT / "docs/site/index.html",
    ROOT / "docs/site/quickstart.html",
]


CLI_SOURCE = ROOT / "crates/kiln-server/src/cli.rs"
CLI_EXAMPLE_SURFACES = [
    ROOT / "README.md",
    ROOT / "QUICKSTART.md",
    ROOT / "docs/site/cli.html",
    *sorted((ROOT / "docs/site").glob("*.html")),
]

CLI_BINARIES = ("kiln", "./target/release/kiln")
NO_VALUE_FLAGS = frozenset({"--json", "--quiet", "-q", "--verbose", "-v"})
ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
DOCS_SITE = ROOT / "docs/site"
DOCS_SITE_URL_ATTR_RE = re.compile(r"\b(?:href|src)\s*=\s*(['\"])(.*?)\1", re.IGNORECASE | re.DOTALL)
DOCS_SITE_JS_URL_RE = re.compile(r"\b(?:cast|script)\s*:\s*(['\"])(.*?)\1")
IGNORED_LOCAL_URL_PREFIXES = ("mailto:", "tel:", "javascript:", "data:")


@dataclass(frozen=True)
class CommandSpec:
    flags: frozenset[str]
    positionals: int = 0


@dataclass(frozen=True)
class CliSurface:
    global_flags: frozenset[str]
    commands: dict[tuple[str, ...], CommandSpec]


def variant_name_to_cli(name: str) -> str:
    words = re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", name)
    return "-".join(word.lower() for word in words)


def matching_brace(text: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise SystemExit("could not parse CLI source: unmatched brace")


def extract_block_after(text: str, marker: str) -> str:
    marker_index = text.find(marker)
    if marker_index < 0:
        raise SystemExit(f"could not parse CLI source: missing {marker!r}")
    open_index = text.find("{", marker_index)
    if open_index < 0:
        raise SystemExit(f"could not parse CLI source: missing block for {marker!r}")
    return text[open_index + 1 : matching_brace(text, open_index)]


def arg_flags(attrs: str, field_name: str) -> set[str]:
    flags: set[str] = set()
    if "long" in attrs:
        long_name = field_name.replace("_", "-")
        explicit_long = re.search(r'long\s*=\s*"([^"]+)"', attrs)
        if explicit_long:
            long_name = explicit_long.group(1)
        flags.add(f"--{long_name}")
    if "short" in attrs:
        short_name = field_name[0]
        explicit_short = re.search(r"short\s*=\s*'([^']+)'", attrs)
        if explicit_short:
            short_name = explicit_short.group(1)
        flags.add(f"-{short_name}")
    return flags


def parse_arg_fields(block: str) -> tuple[set[str], int]:
    flags: set[str] = set()
    positionals = 0
    pending_attrs: list[str] = []

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if line.startswith("#[arg("):
            pending_attrs.append(line)
            continue
        field_match = re.match(r"(?:pub\s+)?([a-z][a-z0-9_]*)\s*:", line)
        if not field_match:
            if line and not line.startswith("///"):
                pending_attrs = []
            continue

        field_name = field_match.group(1)
        attrs = " ".join(pending_attrs)
        pending_attrs = []
        if "command(subcommand)" in attrs:
            continue
        field_flags = arg_flags(attrs, field_name)
        if field_flags:
            flags.update(field_flags)
        else:
            positionals += 1

    return flags, positionals


def parse_variant_blocks(enum_block: str) -> dict[str, str]:
    variants: dict[str, str] = {}
    index = 0
    while index < len(enum_block):
        match = re.search(r"(?m)^\s*(?:#\[[^\n]+\]\s*)*(?:///[^\n]*\n\s*)*([A-Z][A-Za-z0-9]*)\s*\{", enum_block[index:])
        if not match:
            break
        name = match.group(1)
        open_index = index + match.end() - 1
        close_index = matching_brace(enum_block, open_index)
        variants[name] = enum_block[open_index + 1 : close_index]
        index = close_index + 1
    return variants


def parse_cli_surface() -> CliSurface:
    text = CLI_SOURCE.read_text()
    global_flags, _ = parse_arg_fields(extract_block_after(text, "pub struct Cli"))

    command_variants = parse_variant_blocks(extract_block_after(text, "pub enum Commands"))
    train_variants = parse_variant_blocks(extract_block_after(text, "pub enum TrainCommands"))
    adapter_variants = parse_variant_blocks(extract_block_after(text, "pub enum AdapterCommands"))

    commands: dict[tuple[str, ...], CommandSpec] = {}
    for variant, block in command_variants.items():
        if variant in {"Train", "Adapters"}:
            continue
        cli_name = "config" if variant == "ConfigCheck" else variant_name_to_cli(variant)
        flags, positionals = parse_arg_fields(block)
        commands[(cli_name,)] = CommandSpec(frozenset(flags), positionals)

    for variant, block in train_variants.items():
        flags, positionals = parse_arg_fields(block)
        commands[("train", variant_name_to_cli(variant))] = CommandSpec(frozenset(flags), positionals)

    for variant, block in adapter_variants.items():
        flags, positionals = parse_arg_fields(block)
        commands[("adapters", variant_name_to_cli(variant))] = CommandSpec(frozenset(flags), positionals)

    return CliSurface(frozenset(global_flags), commands)


def cli_example_surfaces() -> list[Path]:
    return sorted(set(CLI_EXAMPLE_SURFACES))


def strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def extract_code_examples(text: str, is_html: bool) -> list[tuple[int, str]]:
    examples: list[tuple[int, str]] = []
    if is_html:
        pattern = re.compile(r"<code[^>]*>(.*?)</code>", re.DOTALL)
    else:
        pattern = re.compile(r"```[^\n]*\n(.*?)```|`([^`]+)`", re.DOTALL)

    for match in pattern.finditer(text):
        content = match.group(1) if match.group(1) is not None else match.group(2)
        if is_html:
            content = strip_html_tags(content)
        content = html.unescape(content)
        line = text.count("\n", 0, match.start()) + 1
        examples.append((line, content))
    return examples


def logical_shell_commands(example: str, start_line: int) -> list[tuple[int, str]]:
    commands: list[tuple[int, str]] = []
    pending = ""
    pending_line = start_line
    for offset, raw_line in enumerate(example.splitlines()):
        line_number = start_line + offset
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("$"):
            line = line[1:].strip()
        if pending:
            pending += " " + line
        else:
            pending = line
            pending_line = line_number
        if pending.endswith("\\"):
            pending = pending[:-1].strip()
            continue
        commands.append((pending_line, pending))
        pending = ""
    if pending:
        commands.append((pending_line, pending))
    return commands


def kiln_tokens(command: str) -> list[str] | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    while tokens and ENV_ASSIGNMENT_RE.match(tokens[0]):
        tokens = tokens[1:]
    if "|" in tokens:
        tokens = tokens[: tokens.index("|")]
    if not tokens or tokens[0] not in CLI_BINARIES:
        return None
    if "..." in tokens:
        return None
    return tokens


def flag_name(token: str) -> str | None:
    if token == "--":
        return None
    if token.startswith("--"):
        return token.split("=", 1)[0]
    if re.fullmatch(r"-[A-Za-z]+", token):
        return token
    return None


def short_cluster_flags(token: str) -> list[str] | None:
    if re.fullmatch(r"-[A-Za-z]{2,}", token):
        return [f"-{char}" for char in token[1:]]
    return None


def skip_flag_value(tokens: list[str], index: int) -> int:
    token = tokens[index]
    flag = flag_name(token)
    if flag in NO_VALUE_FLAGS:
        return index + 1
    if token.startswith("--") and "=" in token:
        return index + 1
    if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
        return index + 2
    return index + 1


def command_key(tokens: list[str], surface: CliSurface) -> tuple[tuple[str, ...] | None, int, str | None]:
    index = 1
    while index < len(tokens):
        token = tokens[index]
        flag = flag_name(token)
        cluster = short_cluster_flags(token)
        if cluster:
            unsupported = [short_flag for short_flag in cluster if short_flag not in surface.global_flags]
            if unsupported:
                return None, index, f"unsupported global flag {unsupported[0]!r} before subcommand"
            index += 1
            continue
        if flag and flag in surface.global_flags:
            index = skip_flag_value(tokens, index)
            continue
        if flag:
            return None, index, f"unsupported global flag {flag!r} before subcommand"
        break
    if index >= len(tokens):
        return None, index, None

    first = tokens[index]
    if first in {"train", "adapters"}:
        if index + 1 >= len(tokens):
            return None, index, f"unknown subcommand {' '.join(tokens[1:index + 1])!r}"
        key = (first, tokens[index + 1])
        if key not in surface.commands:
            return None, index + 1, f"unknown subcommand {' '.join(key)!r}"
        return key, index + 2, None

    key = (first,)
    if key not in surface.commands:
        return None, index, f"unknown subcommand {first!r}"
    return key, index + 1, None


def check_cli_command(tokens: list[str], surface: CliSurface) -> str | None:
    key, args_start, key_error = command_key(tokens, surface)
    if key_error:
        return key_error
    if key is None:
        return None

    spec = surface.commands[key]
    allowed_flags = surface.global_flags | spec.flags
    positionals_seen = 0
    index = args_start
    while index < len(tokens):
        token = tokens[index]
        flag = flag_name(token)
        if flag:
            cluster = short_cluster_flags(flag)
            if cluster:
                unsupported = [short_flag for short_flag in cluster if short_flag not in allowed_flags]
                if unsupported:
                    return f"unsupported flag {unsupported[0]!r} for command {' '.join(key)!r}"
                index += 1
                continue
            if flag not in allowed_flags:
                return f"unsupported flag {flag!r} for command {' '.join(key)!r}"
            index = skip_flag_value(tokens, index)
            continue
        positionals_seen += 1
        if positionals_seen > spec.positionals:
            return f"unexpected positional argument {token!r} for command {' '.join(key)!r}"
        index += 1

    return None


def check_cli_examples() -> list[str]:
    errors: list[str] = []
    surface = parse_cli_surface()
    for path in cli_example_surfaces():
        text = path.read_text()
        for start_line, example in extract_code_examples(text, path.suffix == ".html"):
            for line, command in logical_shell_commands(example, start_line):
                tokens = kiln_tokens(command)
                if not tokens:
                    continue
                error = check_cli_command(tokens, surface)
                if error:
                    errors.append(f"{rel(path)}:{line}: {error}: {command!r}")
    return errors


def docs_site_html_pages() -> list[Path]:
    return sorted(DOCS_SITE.glob("**/*.html"))


def is_ignored_docs_site_url(url: str) -> bool:
    stripped = html.unescape(url).strip()
    if not stripped or stripped.startswith("#") or "${" in stripped:
        return True
    lowered = stripped.lower()
    if lowered.startswith(IGNORED_LOCAL_URL_PREFIXES):
        return True
    parsed = urlsplit(stripped)
    return bool(parsed.scheme or parsed.netloc)


def resolve_docs_site_url(source: Path, url: str) -> Path:
    stripped = html.unescape(url).strip()
    parsed = urlsplit(stripped)
    relative_path = unquote(parsed.path)
    base = source.parent / relative_path
    if stripped.endswith("/") or relative_path in {"", ".", ".."}:
        base = base / "index.html"
    return base.resolve()


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def check_docs_site_local_links() -> list[str]:
    errors: list[str] = []
    docs_site_root = DOCS_SITE.resolve()
    for path in docs_site_html_pages():
        text = path.read_text()
        local_reference_matches = [
            *DOCS_SITE_URL_ATTR_RE.finditer(text),
            *DOCS_SITE_JS_URL_RE.finditer(text),
        ]
        for match in sorted(local_reference_matches, key=lambda item: item.start()):
            url = match.group(2)
            if is_ignored_docs_site_url(url):
                continue
            resolved = resolve_docs_site_url(path, url)
            line = text.count("\n", 0, match.start()) + 1
            if not is_relative_to(resolved, docs_site_root):
                errors.append(
                    f"{rel(path)}:{line}: docs site local link escapes docs/site: "
                    f"{url!r} resolves to {resolved}"
                )
                continue
            if not resolved.exists():
                errors.append(
                    f"{rel(path)}:{line}: broken docs site local link {url!r}: "
                    f"missing {rel(resolved)}"
                )
            elif resolved.is_dir():
                errors.append(
                    f"{rel(path)}:{line}: docs site local link {url!r} resolves to directory "
                    f"without trailing index.html: {rel(resolved)}"
                )
    return errors


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    errors: list[str] = []

    for path in CURRENT_SERVER_SURFACES:
        text = path.read_text()
        for pattern, reason in DISALLOWED_CURRENT_SERVER_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                line_text = text.splitlines()[line - 1]
                if "Since kiln-v" in line_text:
                    continue
                errors.append(f"{rel(path)}:{line}: {reason}: {match.group(0)!r}")

    expected_desktop_tag = f"desktop-v{DESKTOP_VERSION}"
    expected_desktop_asset = f"Kiln.Desktop_{DESKTOP_VERSION}_"
    for path in DESKTOP_SURFACES:
        text = path.read_text()
        for match in re.finditer(r"desktop-v[0-9]+\.[0-9]+\.[0-9]+", text):
            if match.group(0) != expected_desktop_tag:
                line = text.count("\n", 0, match.start()) + 1
                errors.append(
                    f"{rel(path)}:{line}: desktop release tag should match "
                    f"desktop/Cargo.toml ({expected_desktop_tag}), got {match.group(0)!r}"
                )
        for match in re.finditer(r"Kiln\.Desktop_[0-9]+\.[0-9]+\.[0-9]+_", text):
            if match.group(0) != expected_desktop_asset:
                line = text.count("\n", 0, match.start()) + 1
                errors.append(
                    f"{rel(path)}:{line}: desktop installer asset should match "
                    f"desktop/Cargo.toml ({expected_desktop_asset}*), got {match.group(0)!r}"
                )

    required_latest_snippets = {
        ROOT / "docs/site/index.html": ["releases/latest", "KILN_VERSION=$(curl -fsSL https://api.github.com/repos/ericflo/kiln/releases/latest"],
        ROOT / "docs/site/quickstart.html": ["KILN_VERSION=$(curl -fsSL https://api.github.com/repos/ericflo/kiln/releases/latest", "ghcr.io/ericflo/kiln-server:latest"],
        ROOT / "README.md": ["ghcr.io/ericflo/kiln-server:latest", "KILN_VERSION=$(curl -fsSL https://api.github.com/repos/ericflo/kiln/releases/latest"],
    }
    for path, snippets in required_latest_snippets.items():
        text = path.read_text()
        for snippet in snippets:
            if snippet not in text:
                errors.append(f"{rel(path)}: missing latest-version snippet {snippet!r}")

    errors.extend(check_cli_examples())
    errors.extend(check_docs_site_local_links())

    if errors:
        print("release version drift check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "release version drift check passed: "
        f"server examples avoid pinned {SERVER_VERSION}; desktop pins match {expected_desktop_tag}; "
        "CLI examples match cli.rs; docs/site local links resolve"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
