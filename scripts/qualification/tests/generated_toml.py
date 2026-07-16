from __future__ import annotations

import json


def parse_generated_toml(source: str) -> dict[str, dict[str, object]]:
    """Parse the closed JSON-scalar TOML subset emitted by qualification drivers."""
    parsed: dict[str, dict[str, object]] = {}
    section: dict[str, object] | None = None
    for line_number, raw in enumerate(source.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            if not name or name in parsed:
                raise AssertionError(
                    f"invalid generated TOML section on line {line_number}"
                )
            section = {}
            parsed[name] = section
            continue
        if section is None or "=" not in line:
            raise AssertionError(
                f"invalid generated TOML scalar on line {line_number}"
            )
        key, raw_value = (part.strip() for part in line.split("=", 1))
        if not key or key in section:
            raise AssertionError(
                f"duplicate generated TOML key on line {line_number}"
            )
        section[key] = json.loads(raw_value)
    return parsed
