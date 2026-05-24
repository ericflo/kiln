#!/usr/bin/env python3
"""Prepare postbreach H1 idealized SFT traces from train tasks only."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent

BASE_SYSTEM_PROMPT = (
    "You are a coding assistant with read, bash, edit, and write tools. "
    "Read the relevant file before editing, make the requested change, "
    "verify it, and then send a final text response."
)

STYLE_SUMMARY = {
    "py_strict_typed_snake": "snake_case naming, strict type annotations, logging, try/except handling, and docstrings",
    "py_camel_loose": "camelCase naming, no type annotations, print-style diagnostics, inline comments, and raise-only handling",
    "rust_snake_result": "snake_case naming, Result return style, strict types, and doc comments",
    "js_camel_warn_jsdoc": "camelCase naming, JSDoc comments, console.warn diagnostics, throw Error handling, and CommonJS exports",
}

THINK_VARIANTS = [
    (
        "I will read the target file before editing so the new code matches the local style.",
        "The file shows the naming, error, and comment conventions. I will make the smallest full-file edit.",
        "The edit is written; I will run the verifier before finishing.",
        "The verifier passed, so I will answer with the file and preserved conventions.",
    ),
    (
        "First I need the nearby implementation and style.",
        "The requested addition can follow the existing function shape without unrelated changes.",
        "Now I will validate the modified workspace.",
        "Validation passed; the final response should be one concise sentence.",
    ),
    (
        "I should inspect the target file before any write.",
        "The conventions are clear from the existing code, so I will add only the requested function.",
        "Next I will run the provided check.",
        "The check passed, so I can finish with the required summary.",
    ),
]


def tool_call(name: str, arguments: dict) -> str:
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=True)}</tool_call>"


def _strict_new(task_id: str) -> str:
    wants_millis = "001" in task_id or "003" in task_id or "005" in task_id or "007" in task_id
    extra = (
        "def parse_unix_millis(timestamp: int) -> dt.datetime:\n"
        '    """Parse Unix milliseconds as a UTC datetime."""\n'
        "    try:\n"
        "        return dt.datetime.fromtimestamp(timestamp / 1000, tz=dt.timezone.utc)\n"
        "    except (OSError, OverflowError, ValueError):\n"
        '        logger.exception("failed to parse unix milliseconds %r", timestamp)\n'
        "        raise\n"
        if wants_millis
        else
        "def parse_unix(timestamp: int) -> dt.datetime:\n"
        '    """Parse Unix seconds as a UTC datetime."""\n'
        "    try:\n"
        "        return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)\n"
        "    except (OSError, OverflowError, ValueError):\n"
        '        logger.exception("failed to parse unix timestamp %r", timestamp)\n'
        "        raise\n"
    )
    return (
        '"""Date parsing helpers."""\n'
        "from __future__ import annotations\n"
        "import datetime as dt\n"
        "import logging\n\n"
        "logger = logging.getLogger(__name__)\n\n\n"
        "def parse_iso(s: str) -> dt.datetime:\n"
        '    """Parse an ISO 8601 timestamp string."""\n'
        "    try:\n"
        "        return dt.datetime.fromisoformat(s)\n"
        "    except ValueError:\n"
        '        logger.exception("failed to parse %r", s)\n'
        "        raise\n\n\n"
        f"{extra}"
    )


def _loose_new(task_id: str) -> str:
    wants_millis = "001" in task_id or "003" in task_id or "005" in task_id or "007" in task_id
    extra = (
        "def parseUnixMillis(ts):\n"
        "    # parse unix milliseconds\n"
        "    if ts is None:\n"
        "        print('failed to parse', ts)\n"
        "        raise ValueError('missing timestamp')\n"
        "    return datetime.datetime.utcfromtimestamp(ts / 1000.0)\n"
        if wants_millis
        else
        "def parseUnix(ts):\n"
        "    # parse unix timestamp\n"
        "    if ts is None:\n"
        "        print('failed to parse', ts)\n"
        "        raise ValueError('missing timestamp')\n"
        "    return datetime.datetime.utcfromtimestamp(ts)\n"
    )
    return (
        "import datetime\n\n"
        "def parseIso(s):\n"
        "    # parse iso string\n"
        "    try:\n"
        "        return datetime.datetime.fromisoformat(s)\n"
        "    except:\n"
        "        print('failed to parse', s)\n"
        "        raise\n\n\n"
        f"{extra}"
    )


def _rust_new(task_id: str) -> str:
    wants_minutes = "001" in task_id or "003" in task_id or "005" in task_id or "007" in task_id
    extra = (
        "/// Parse a decimal minute duration.\n"
        "pub fn parse_minutes(s: &str) -> Result<Duration, ParseIntError> {\n"
        "    let minutes: u64 = s.parse()?;\n"
        "    Ok(Duration::from_secs(minutes * 60))\n"
        "}\n"
        if wants_minutes
        else
        "/// Parse a decimal second duration.\n"
        "pub fn parse_seconds(s: &str) -> Result<Duration, ParseIntError> {\n"
        "    let seconds: u64 = s.parse()?;\n"
        "    Ok(Duration::from_secs(seconds))\n"
        "}\n"
    )
    return (
        "use std::num::ParseIntError;\n"
        "use std::time::Duration;\n\n"
        "/// Parse a decimal millisecond duration.\n"
        "pub fn parse_millis(s: &str) -> Result<Duration, ParseIntError> {\n"
        "    let millis: u64 = s.parse()?;\n"
        "    Ok(Duration::from_millis(millis))\n"
        "}\n\n"
        f"{extra}"
    )


def _js_new(task_id: str) -> str:
    wants_display = "001" in task_id or "003" in task_id or "005" in task_id or "007" in task_id
    extra = (
        "/**\n"
        " * Format a name as Last, First.\n"
        " */\n"
        "function formatDisplayName(firstName, lastName) {\n"
        "  if (!firstName || !lastName) {\n"
        "    console.warn('missing name part');\n"
        "    throw new Error('missing name part');\n"
        "  }\n"
        "  return `${lastName.trim()}, ${firstName.trim()}`;\n"
        "}\n"
        if wants_display
        else
        "/**\n"
        " * Format initials for a first and last name.\n"
        " */\n"
        "function formatInitials(firstName, lastName) {\n"
        "  if (!firstName || !lastName) {\n"
        "    console.warn('missing name part');\n"
        "    throw new Error('missing name part');\n"
        "  }\n"
        "  return `${firstName.trim()[0]}.${lastName.trim()[0]}.`;\n"
        "}\n"
    )
    export_name = "formatDisplayName" if wants_display else "formatInitials"
    return (
        "/**\n"
        " * Format a first and last name for display.\n"
        " */\n"
        "function formatName(firstName, lastName) {\n"
        "  if (!firstName || !lastName) {\n"
        "    console.warn('missing name part');\n"
        "    throw new Error('missing name part');\n"
        "  }\n"
        "  return `${firstName.trim()} ${lastName.trim()}`;\n"
        "}\n\n"
        f"{extra}\n"
        f"module.exports = {{ formatName, {export_name} }};\n"
    )


NEW_CONTENT = {
    "py_strict_typed_snake": _strict_new,
    "py_camel_loose": _loose_new,
    "rust_snake_result": _rust_new,
    "js_camel_warn_jsdoc": _js_new,
}


def load_system_prompt(config_path: Path) -> str:
    cfg = json.loads(config_path.read_text())
    append = cfg.get("pi_append_system_prompt")
    if append:
        return f"{BASE_SYSTEM_PROMPT}\n\n{append}"
    return BASE_SYSTEM_PROMPT


def verify_new_content(task: dict, target_path: str, new_content: str) -> None:
    with tempfile.TemporaryDirectory(prefix="pi-context-pb-h1-verify-") as td:
        root = Path(td)
        for rel, content in (task.get("init_files") or {}).items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        (root / target_path).write_text(new_content)
        env = dict(os.environ)
        env["PATH"] = f"{Path.home() / '.cargo' / 'bin'}:{env.get('PATH', '')}"
        result = subprocess.run(
            ["bash", "-c", task["verify_cmd"]],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"verify failed for {task['task_id']}:\n"
                f"stdout={result.stdout}\nstderr={result.stderr}"
            )


def assistant_text(task: dict, target_path: str, old: str, new: str, variant_idx: int) -> str:
    profile = task["profile"]["name"]
    t1, t2, t3, t4 = THINK_VARIANTS[variant_idx % len(THINK_VARIANTS)]
    verify_cmd = task["verify_cmd"]
    final = f"Modified {target_path}; preserved {STYLE_SUMMARY[profile]}."
    return "<TURN_BREAK>".join(
        [
            f"<think>{t1}</think>" + tool_call("read", {"path": target_path}),
            f"<think>{t2}</think>" + tool_call("edit", {"path": target_path, "edits": [{"oldText": old, "newText": new}]}),
            f"<think>{t3}</think>" + tool_call("bash", {"command": verify_cmd}),
            f"<think>{t4}</think>{final}",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(ROOT / "datasets/train.tasks.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "datasets/sft.pb-h1-ideal-traces.jsonl"))
    ap.add_argument("--config", default=str(ROOT / "capability.config.json"))
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    system_prompt = load_system_prompt(Path(args.config))
    rows = []
    profile_counts: dict[str, int] = defaultdict(int)
    for idx, line in enumerate(Path(args.tasks).read_text().splitlines()):
        if not line.strip():
            continue
        task = json.loads(line)
        profile = task["profile"]["name"]
        if profile not in NEW_CONTENT:
            continue
        init_files = task.get("init_files") or {}
        if len(init_files) != 1:
            raise ValueError(f"expected one target file for {task['task_id']}")
        target_path, old_content = next(iter(init_files.items()))
        new_content = NEW_CONTENT[profile](task["task_id"])
        if not args.skip_verify:
            verify_new_content(task, target_path, new_content)
        completion = assistant_text(task, target_path, old_content, new_content, idx)
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task["prompt"]},
                    {"role": "assistant", "content": completion},
                ],
                "_meta": {
                    "task_id": task["task_id"],
                    "profile": profile,
                    "target_path": target_path,
                    "completion_chars": len(completion),
                    "verified": not args.skip_verify,
                },
            }
        )
        profile_counts[profile] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    lengths = [row["_meta"]["completion_chars"] for row in rows]
    print(f"wrote={out} examples={len(rows)} profiles={dict(sorted(profile_counts.items()))}")
    if lengths:
        print(f"completion_chars_min={min(lengths)} completion_chars_max={max(lengths)}")


if __name__ == "__main__":
    main()
