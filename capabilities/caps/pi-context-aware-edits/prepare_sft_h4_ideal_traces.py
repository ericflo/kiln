#!/usr/bin/env python3
"""Prepare H4 idealized train-only SFT traces.

This script reads train tasks only. It does not read eval tasks or any eval
rollout transcripts. The generated examples are deterministic, verifier-backed
read/edit/verify/final traces in the same flattened action-text shape used by
the Pi trajectory normalizer.
"""

from __future__ import annotations

import argparse
import json
import shutil
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


THINK_VARIANTS = [
    (
        "I will read the target file first so the edit follows the local style.",
        "The file shows the conventions I need to preserve. I will make the requested change with the smallest full-file edit.",
        "Now I will run the provided verifier before sending the final response.",
        "The verifier passed, so I can give the required final sentence.",
    ),
    (
        "First I need the existing implementation and nearby style before editing.",
        "The conventions are clear from the file. I will add only the requested function and keep the surrounding style intact.",
        "The edit is in place; I will verify it with the task command.",
        "Verification succeeded, so the final answer should be concise.",
    ),
    (
        "I should inspect the target file before changing it.",
        "I found the naming, typing, and comment pattern. I will apply a matching edit.",
        "Next I will run the verifier to confirm the file still works.",
        "The check passed, so I will name the file and preserved conventions.",
    ),
    (
        "I will gather local context from the target file before any write.",
        "The requested addition can be made without changing unrelated code.",
        "I will run the validation command after the edit.",
        "Validation passed; now I can finish with one sentence.",
    ),
]


def strict_python_new_content() -> str:
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
        "def parse_unix(timestamp: int) -> dt.datetime:\n"
        '    """Parse a Unix timestamp as a UTC datetime."""\n'
        "    try:\n"
        "        return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)\n"
        "    except (OSError, OverflowError, ValueError):\n"
        '        logger.exception("failed to parse unix timestamp %r", timestamp)\n'
        "        raise\n"
    )


def loose_python_new_content() -> str:
    return (
        "import datetime\n\n"
        "def parseIso(s):\n"
        "    # parse iso string\n"
        "    try:\n"
        "        return datetime.datetime.fromisoformat(s)\n"
        "    except:\n"
        "        print('failed to parse', s)\n"
        "        raise\n\n\n"
        "def parseUnix(ts):\n"
        "    # parse unix timestamp\n"
        "    if ts is None:\n"
        "        print('failed to parse', ts)\n"
        "        raise ValueError('missing timestamp')\n"
        "    return datetime.datetime.fromtimestamp(ts)\n"
    )


def rust_new_content() -> str:
    return (
        "use std::num::ParseIntError;\n"
        "use std::time::Duration;\n\n"
        "/// Parse a decimal millisecond duration.\n"
        "pub fn parse_millis(s: &str) -> Result<Duration, ParseIntError> {\n"
        "    let millis: u64 = s.parse()?;\n"
        "    Ok(Duration::from_millis(millis))\n"
        "}\n\n"
        "/// Parse a decimal second duration.\n"
        "pub fn parse_seconds(s: &str) -> Result<Duration, ParseIntError> {\n"
        "    let seconds: u64 = s.parse()?;\n"
        "    Ok(Duration::from_secs(seconds))\n"
        "}\n"
    )


NEW_CONTENT = {
    "py_strict_typed_snake": strict_python_new_content,
    "py_camel_loose": loose_python_new_content,
    "rust_snake_result": rust_new_content,
}


STYLE_SUMMARY = {
    "py_strict_typed_snake": "snake_case naming, strict type annotations, logging, try/except handling, and docstrings",
    "py_camel_loose": "camelCase naming, no type annotations, print-style diagnostics, inline comments, and raise-only handling",
    "rust_snake_result": "snake_case naming, Result return style, strict types, and doc comments",
}


def tool_call(name: str, arguments: dict) -> str:
    payload = {"name": name, "arguments": arguments}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=True)}</tool_call>"


def assistant_text(task: dict, target_path: str, old: str, new: str, variant_idx: int) -> str:
    profile = task["profile"]["name"]
    t1, t2, t3, t4 = THINK_VARIANTS[variant_idx % len(THINK_VARIANTS)]
    verify_cmd = task["verify_cmd"]
    final = f"Modified {target_path}; preserved {STYLE_SUMMARY[profile]}."
    parts = [
        f"<think>{t1}</think>" + tool_call("read", {"path": target_path}),
        f"<think>{t2}</think>"
        + tool_call(
            "edit",
            {
                "path": target_path,
                "edits": [
                    {
                        "oldText": old,
                        "newText": new,
                    }
                ],
            },
        ),
        f"<think>{t3}</think>" + tool_call("bash", {"command": verify_cmd}),
        f"<think>{t4}</think>{final}",
    ]
    return "<TURN_BREAK>".join(parts)


def verify_new_content(task: dict, target_path: str, new_content: str) -> None:
    with tempfile.TemporaryDirectory(prefix="pi-context-h4-verify-") as td:
        root = Path(td)
        for rel, content in (task.get("init_files") or {}).items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        (root / target_path).write_text(new_content)
        result = subprocess.run(
            ["bash", "-c", task["verify_cmd"]],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"verify failed for {task['task_id']}:\n"
                f"stdout={result.stdout}\nstderr={result.stderr}"
            )


def load_system_prompt(config_path: Path) -> str:
    cfg = json.loads(config_path.read_text())
    append = cfg.get("pi_append_system_prompt")
    if append:
        return f"{BASE_SYSTEM_PROMPT}\n\n{append}"
    return BASE_SYSTEM_PROMPT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(ROOT / "datasets/train.tasks.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "datasets/sft.h4-ideal-traces.jsonl"))
    ap.add_argument("--config", default=str(ROOT / "capability.config.json"))
    ap.add_argument("--per-profile", type=int, default=4)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    tasks_by_profile: dict[str, list[dict]] = defaultdict(list)
    for line in Path(args.tasks).read_text().splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        profile = task["profile"]["name"]
        if profile in NEW_CONTENT:
            tasks_by_profile[profile].append(task)

    system_prompt = load_system_prompt(Path(args.config))
    rows = []
    for profile in sorted(tasks_by_profile):
        for variant_idx, task in enumerate(tasks_by_profile[profile][: args.per_profile]):
            init_files = task.get("init_files") or {}
            if len(init_files) != 1:
                raise ValueError(f"expected one target file for {task['task_id']}")
            target_path, old_content = next(iter(init_files.items()))
            new_content = NEW_CONTENT[profile]()
            if not args.skip_verify:
                verify_new_content(task, target_path, new_content)
            completion = assistant_text(
                task,
                target_path,
                old_content,
                new_content,
                variant_idx,
            )
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    lengths = [row["_meta"]["completion_chars"] for row in rows]
    profiles = defaultdict(int)
    for row in rows:
        profiles[row["_meta"]["profile"]] += 1
    print(f"wrote={out} examples={len(rows)} profiles={dict(sorted(profiles.items()))}")
    if lengths:
        print(f"chars_min={min(lengths)} chars_max={max(lengths)}")
    if shutil.which("rustc") is None:
        print("warning=rustc not found during script runtime")


if __name__ == "__main__":
    main()
