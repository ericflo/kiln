#!/usr/bin/env python3
"""Guard user-facing Kiln release examples against stale version literals."""
from __future__ import annotations

import re
import sys
from pathlib import Path

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

    if errors:
        print("release version drift check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "release version drift check passed: "
        f"server examples avoid pinned {SERVER_VERSION}; desktop pins match {expected_desktop_tag}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
