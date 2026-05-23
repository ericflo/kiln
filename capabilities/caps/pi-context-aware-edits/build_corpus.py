"""Task-corpus builder for pi-context-aware-edits.

Each task is a small workspace with a *consistent style* across all
files, plus an edit request. The rubric measures whether the agent's
edit preserves the style.

Convention categories tracked:
  1. import_style    — top alphabetical vs grouped
  2. naming_case     — snake_case / camelCase / PascalCase
  3. error_handling  — try/except, Result, panic, unwrap
  4. logging_style   — print, logging.X, structured
  5. comment_style   — docstrings, inline, minimal
  6. type_annotations — present / absent / partial / strict

Output:
  datasets/train.tasks.jsonl  (30 tasks across mixed conventions)
  datasets/eval.tasks.jsonl   (18 held-out tasks)
"""
from __future__ import annotations
import json
import os
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SEED = int(os.environ.get("SEED", 31417))

FINAL_RESPONSE_CONTRACT = (
    "After editing and verifying, reply with one concise sentence naming the "
    "modified file and the style convention(s) you preserved."
)


def _prompt(text: str) -> str:
    return f"{text}\n\n{FINAL_RESPONSE_CONTRACT}"


# Style profiles per workspace. Each profile names a consistent convention
# across all files.
PROFILES = [
    {
        "name": "py_strict_typed_snake",
        "language": "python",
        "naming_case": "snake_case",
        "type_annotations": "strict",
        "logging_style": "logging",
        "comment_style": "docstrings",
        "import_style": "top_alphabetical",
        "error_handling": "try_except",
    },
    {
        "name": "py_camel_loose",
        "language": "python",
        "naming_case": "camelCase",
        "type_annotations": "absent",
        "logging_style": "print",
        "comment_style": "inline",
        "import_style": "top_grouped",
        "error_handling": "raise_only",
    },
    {
        "name": "rust_snake_result",
        "language": "rust",
        "naming_case": "snake_case",
        "type_annotations": "strict",
        "logging_style": "log_crate",
        "comment_style": "docstrings",
        "import_style": "top_alphabetical",
        "error_handling": "result",
    },
    {
        "name": "go_camel_pascal",
        "language": "go",
        "naming_case": "PascalCase_pub_camel_priv",
        "type_annotations": "strict",
        "logging_style": "log_package",
        "comment_style": "godoc",
        "import_style": "grouped",
        "error_handling": "explicit_err",
    },
]


def _t_add_function_python_strict(profile, idx) -> dict:
    return {
        "task_id": f"{profile['name']}_add_fn_{idx:03d}",
        "profile": profile,
        "init_files": {
            "lib/parse.py": (
                "\"\"\"Date parsing helpers.\"\"\"\n"
                "from __future__ import annotations\n"
                "import datetime as dt\n"
                "import logging\n\n"
                "logger = logging.getLogger(__name__)\n\n\n"
                "def parse_iso(s: str) -> dt.datetime:\n"
                "    \"\"\"Parse an ISO 8601 timestamp string.\"\"\"\n"
                "    try:\n"
                "        return dt.datetime.fromisoformat(s)\n"
                "    except ValueError:\n"
                "        logger.exception(\"failed to parse %r\", s)\n"
                "        raise\n"
            ),
        },
        "prompt": _prompt(
            "Add a `parse_unix(timestamp: int) -> datetime.datetime` function "
            "to `lib/parse.py`. It converts a Unix timestamp (seconds since "
            "epoch) to a UTC datetime. Match the existing style of this file."
        ),
        "expected_conventions_in_edit": {
            "naming_case": "snake_case",
            "type_annotations": "strict",
            "logging_style": "logging",
            "error_handling": "try_except",
        },
        "gold_state_predicate": "import + function callable",
        "verify_cmd": "python3 -c 'from lib.parse import parse_unix; import datetime; assert parse_unix(0).year == 1970'",
    }


def _t_add_function_python_loose(profile, idx) -> dict:
    return {
        "task_id": f"{profile['name']}_add_fn_{idx:03d}",
        "profile": profile,
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
            ),
        },
        "prompt": _prompt(
            "Add a `parseUnix(ts)` function to `lib/parse.py`. It should "
            "convert a Unix timestamp (seconds since epoch) to a datetime. "
            "Match the existing style."
        ),
        "expected_conventions_in_edit": {
            "naming_case": "camelCase",
            "type_annotations": "absent",
            "logging_style": "print",
            "error_handling": "raise_only",
        },
        "gold_state_predicate": "function callable",
        "verify_cmd": "python3 -c 'from lib.parse import parseUnix; import datetime; assert parseUnix(0).year == 1970'",
    }


def _t_add_function_rust(profile, idx) -> dict:
    return {
        "task_id": f"{profile['name']}_add_fn_{idx:03d}",
        "profile": profile,
        "init_files": {
            "src/parse.rs": (
                "use std::num::ParseIntError;\n"
                "use std::time::Duration;\n\n"
                "/// Parse a decimal millisecond duration.\n"
                "pub fn parse_millis(s: &str) -> Result<Duration, ParseIntError> {\n"
                "    let millis: u64 = s.parse()?;\n"
                "    Ok(Duration::from_millis(millis))\n"
                "}\n"
            ),
        },
        "prompt": _prompt(
            "Add a `parse_seconds(s: &str) -> Result<Duration, ParseIntError>` "
            "function to `src/parse.rs`. Match the existing style "
            "(snake_case, Result return, doc comment)."
        ),
        "expected_conventions_in_edit": {
            "naming_case": "snake_case",
            "type_annotations": "strict",
            "error_handling": "result",
            "comment_style": "docstrings",
        },
        "gold_state_predicate": "compiles",
        "verify_cmd": (
            "grep -q 'pub fn parse_seconds' src/parse.rs "
            "&& grep -q 'Result<Duration, ParseIntError>' src/parse.rs "
            "&& rustc --crate-type lib src/parse.rs -o /tmp/pi_context_parse_rs_check.rlib"
        ),
    }


def _t_add_function_go(profile, idx) -> dict:
    return {
        "task_id": f"{profile['name']}_add_fn_{idx:03d}",
        "profile": profile,
        "init_files": {
            "parse.go": (
                "package parse\n\n"
                "import (\n\t\"time\"\n)\n\n"
                "// ParseISO parses an ISO 8601 timestamp string.\n"
                "func ParseISO(s string) (time.Time, error) {\n"
                "\tt, err := time.Parse(time.RFC3339, s)\n"
                "\tif err != nil {\n"
                "\t\treturn time.Time{}, err\n"
                "\t}\n"
                "\treturn t, nil\n"
                "}\n"
            ),
        },
        "prompt": _prompt(
            "Add a `ParseUnix(ts int64) (time.Time, error)` function to "
            "`parse.go`. Match the existing Go style (PascalCase exported, "
            "explicit error returns, godoc comment)."
        ),
        "expected_conventions_in_edit": {
            "naming_case": "PascalCase_pub_camel_priv",
            "error_handling": "explicit_err",
            "comment_style": "godoc",
        },
        "gold_state_predicate": "compiles",
        "verify_cmd": (
            "grep -q 'func ParseUnix' parse.go "
            "&& grep -q 'time.Unix' parse.go "
            "&& GO111MODULE=off go test ./..."
        ),
    }


GENERATORS = {
    "py_strict_typed_snake": _t_add_function_python_strict,
    "py_camel_loose": _t_add_function_python_loose,
    "rust_snake_result": _t_add_function_rust,
    "go_camel_pascal": _t_add_function_go,
}


def main():
    DATASETS.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    train, eval_ = [], []
    available_profiles = []
    for profile in PROFILES:
        language = profile["language"]
        if language == "rust" and shutil.which("rustc") is None:
            print("skip rust profile: rustc is not on PATH")
            continue
        if language == "go" and shutil.which("go") is None:
            print("skip go profile: go is not on PATH")
            continue
        available_profiles.append(profile)
    for profile in available_profiles:
        gen = GENERATORS[profile["name"]]
        for i in range(8):
            train.append(gen(profile, i))
        for i in range(4):
            eval_.append(gen(profile, 100 + i))
    rng.shuffle(train)
    rng.shuffle(eval_)
    with open(DATASETS / "train.tasks.jsonl", "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")
    with open(DATASETS / "eval.tasks.jsonl", "w") as f:
        for t in eval_:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(train)} train tasks")
    print(f"wrote {len(eval_)} eval tasks")


if __name__ == "__main__":
    main()
