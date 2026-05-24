"""Fresh postbreach corpus builder for pi-context-aware-edits.

This generator intentionally does not print task prompts or held-out task
contents. It writes train/eval/hard_eval splits from templates and reports
only aggregate counts, so subsequent training work can keep the eval pool blind.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SEED = int(os.environ.get("SEED", "20260524"))
RUSTC = shutil.which("rustc") or str(Path.home() / ".cargo" / "bin" / "rustc")
NODE = shutil.which("node")

FINAL_RESPONSE_CONTRACT = (
    "After editing and verifying, reply with one concise sentence naming the "
    "modified file and the style convention(s) you preserved."
)


def _prompt(text: str) -> str:
    return f"{text}\n\n{FINAL_RESPONSE_CONTRACT}"


def _strict_parse_task(idx: int, split: str) -> dict:
    fn = "parse_unix_millis" if idx % 2 else "parse_unix"
    verify = (
        "python3 -c 'from lib.parse import parse_unix; "
        "import datetime as dt; assert parse_unix(0) == dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)'"
        if fn == "parse_unix"
        else "python3 -c 'from lib.parse import parse_unix_millis; "
        "import datetime as dt; assert parse_unix_millis(1000) == dt.datetime(1970, 1, 1, 0, 0, 1, tzinfo=dt.timezone.utc)'"
    )
    request = (
        "Add a `parse_unix(timestamp: int) -> dt.datetime` function to `lib/parse.py`. "
        "It converts Unix seconds to a UTC datetime. Match the existing style."
        if fn == "parse_unix"
        else "Add a `parse_unix_millis(timestamp: int) -> dt.datetime` function to `lib/parse.py`. "
        "It converts Unix milliseconds to a UTC datetime. Match the existing style."
    )
    return {
        "task_id": f"{split}_py_strict_parse_{idx:03d}",
        "profile": {
            "name": "py_strict_typed_snake",
            "language": "python",
            "naming_case": "snake_case",
            "type_annotations": "strict",
            "logging_style": "logging",
            "comment_style": "docstrings",
            "import_style": "top_alphabetical",
            "error_handling": "try_except",
        },
        "init_files": {
            "lib/parse.py": (
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
                "        raise\n"
            )
        },
        "prompt": _prompt(request),
        "expected_conventions_in_edit": {
            "naming_case": "snake_case",
            "type_annotations": "strict",
            "logging_style": "logging",
            "error_handling": "try_except",
            "comment_style": "docstrings",
        },
        "gold_state_predicate": "import + function callable",
        "verify_cmd": verify,
    }


def _loose_parse_task(idx: int, split: str) -> dict:
    fn = "parseUnix" if idx % 2 == 0 else "parseUnixMillis"
    verify = (
        "python3 -c 'from lib.parse import parseUnix; import datetime; assert parseUnix(0).year == 1970'"
        if fn == "parseUnix"
        else "python3 -c 'from lib.parse import parseUnixMillis; import datetime; assert parseUnixMillis(1000).year == 1970'"
    )
    request = (
        "Add a `parseUnix(ts)` function to `lib/parse.py`. It converts Unix seconds to a datetime. "
        "Match the existing style."
        if fn == "parseUnix"
        else "Add a `parseUnixMillis(ts)` function to `lib/parse.py`. It converts Unix milliseconds to a datetime. "
        "Match the existing style."
    )
    return {
        "task_id": f"{split}_py_camel_loose_{idx:03d}",
        "profile": {
            "name": "py_camel_loose",
            "language": "python",
            "naming_case": "camelCase",
            "type_annotations": "absent",
            "logging_style": "print",
            "comment_style": "inline",
            "import_style": "top_grouped",
            "error_handling": "raise_only",
        },
        "init_files": {
            "lib/parse.py": (
                "import datetime\n\n"
                "def parseIso(s):\n"
                "    # parse iso string\n"
                "    try:\n"
                "        return datetime.datetime.fromisoformat(s)\n"
                "    except:\n"
                "        print('failed to parse', s)\n"
                "        raise\n"
            )
        },
        "prompt": _prompt(request),
        "expected_conventions_in_edit": {
            "naming_case": "camelCase",
            "type_annotations": "absent",
            "logging_style": "print",
            "error_handling": "raise_only",
            "comment_style": "inline",
        },
        "gold_state_predicate": "function callable",
        "verify_cmd": verify,
    }


def _rust_duration_task(idx: int, split: str) -> dict:
    fn = "parse_seconds" if idx % 2 == 0 else "parse_minutes"
    ctor = "from_secs" if fn == "parse_seconds" else "from_secs(minutes * 60)"
    body = (
        "Add a `parse_seconds(s: &str) -> Result<Duration, ParseIntError>` function to `src/parse.rs`. "
        "Match the existing style (snake_case, Result return, doc comment)."
        if fn == "parse_seconds"
        else "Add a `parse_minutes(s: &str) -> Result<Duration, ParseIntError>` function to `src/parse.rs`. "
        "It should parse decimal minutes and return a Duration. Match the existing style."
    )
    grep_expr = (
        "grep -q 'pub fn parse_seconds' src/parse.rs && grep -q 'Duration::from_secs' src/parse.rs"
        if fn == "parse_seconds"
        else "grep -q 'pub fn parse_minutes' src/parse.rs && grep -q 'minutes \\* 60' src/parse.rs"
    )
    return {
        "task_id": f"{split}_rust_duration_{idx:03d}",
        "profile": {
            "name": "rust_snake_result",
            "language": "rust",
            "naming_case": "snake_case",
            "type_annotations": "strict",
            "logging_style": "log_crate",
            "comment_style": "docstrings",
            "import_style": "top_alphabetical",
            "error_handling": "result",
        },
        "init_files": {
            "src/parse.rs": (
                "use std::num::ParseIntError;\n"
                "use std::time::Duration;\n\n"
                "/// Parse a decimal millisecond duration.\n"
                "pub fn parse_millis(s: &str) -> Result<Duration, ParseIntError> {\n"
                "    let millis: u64 = s.parse()?;\n"
                "    Ok(Duration::from_millis(millis))\n"
                "}\n"
            )
        },
        "prompt": _prompt(body),
        "expected_conventions_in_edit": {
            "naming_case": "snake_case",
            "type_annotations": "strict",
            "error_handling": "result",
            "comment_style": "docstrings",
        },
        "gold_state_predicate": "compiles",
        "verify_cmd": f"{grep_expr} && {RUSTC} --crate-type lib src/parse.rs -o /tmp/pi_context_postbreach_parse_rs_check.rlib",
    }


def _js_format_task(idx: int, split: str) -> dict:
    fn = "formatInitials" if idx % 2 == 0 else "formatDisplayName"
    request = (
        "Add a `formatInitials(firstName, lastName)` function to `src/format.js`. "
        "It should return initials like `A.L.`. Match the existing style."
        if fn == "formatInitials"
        else "Add a `formatDisplayName(firstName, lastName)` function to `src/format.js`. "
        "It should return `Last, First`. Match the existing style."
    )
    verify = (
        "node -e \"const f=require('./src/format.js'); if (f.formatInitials('Ada','Lovelace') !== 'A.L.') process.exit(1)\""
        if fn == "formatInitials"
        else "node -e \"const f=require('./src/format.js'); if (f.formatDisplayName('Ada','Lovelace') !== 'Lovelace, Ada') process.exit(1)\""
    )
    return {
        "task_id": f"{split}_js_camel_warn_{idx:03d}",
        "profile": {
            "name": "js_camel_warn_jsdoc",
            "language": "javascript",
            "naming_case": "camelCase",
            "type_annotations": "absent",
            "logging_style": "console_warn",
            "comment_style": "jsdoc",
            "import_style": "commonjs_exports",
            "error_handling": "throw_error",
        },
        "init_files": {
            "src/format.js": (
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
                "module.exports = { formatName };\n"
            )
        },
        "prompt": _prompt(request),
        "expected_conventions_in_edit": {
            "naming_case": "camelCase",
            "type_annotations": "absent",
            "logging_style": "console_warn",
            "error_handling": "throw_error",
            "comment_style": "jsdoc",
        },
        "gold_state_predicate": "node require callable",
        "verify_cmd": verify,
    }


def _generators() -> list:
    gens = [_strict_parse_task, _loose_parse_task]
    if Path(RUSTC).exists():
        gens.append(_rust_duration_task)
    if NODE:
        gens.append(_js_format_task)
    return gens


def _make_split(split: str, per_profile: int, rng: random.Random) -> list[dict]:
    tasks = []
    for gen in _generators():
        for i in range(per_profile):
            tasks.append(gen(i, split))
    rng.shuffle(tasks)
    return tasks


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    DATASETS.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    train = _make_split("train", 8, rng)
    eval_rows = _make_split("eval", 4, rng)
    hard_eval = _make_split("hard", 3, rng)
    _write_jsonl(DATASETS / "train.tasks.jsonl", train)
    _write_jsonl(DATASETS / "eval.tasks.jsonl", eval_rows)
    _write_jsonl(DATASETS / "hard_eval.tasks.jsonl", hard_eval)
    profiles = sorted({t["profile"]["name"] for t in train + eval_rows + hard_eval})
    print(f"wrote train={len(train)} eval={len(eval_rows)} hard_eval={len(hard_eval)} profiles={','.join(profiles)}")


if __name__ == "__main__":
    main()
